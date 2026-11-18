"""Apply dense, sparse and normalized edits at the selected prediction positions."""

from __future__ import annotations

import types

import torch

from sparse_steering.baselines.dense import realize_dense
from sparse_steering.method.global_normalization import realize_global_normalization
from sparse_steering.method.sparse_edit import realize_batch


def make_additive_hook(self, direction, target_norm, *, matched=True, signs=None):
    records = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        current = hidden[:, -1, :]
        if matched:
            updated, squared_edit_norm, error, cosine = realize_batch(
                current, direction, target_norm
            )
            records.append(torch.stack([squared_edit_norm, error, cosine], dim=-1))
        else:
            delta = target_norm * direction[None, :]
            if signs is not None:
                delta = delta * signs[:, None]
            updated = current + delta.to(current.dtype)
        changed = hidden.clone()
        changed[:, -1, :] = updated
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return hook, records


def make_dense_hook(self, direction, target_norm=0, *, matched=True, signs=None):
    if not matched:
        return make_additive_hook(self, direction, target_norm, matched=False, signs=signs)
    records = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h = hidden[:, -1, :]
        squared_edit_norm_limit = (
            self.cell["edit_norm_ratio_squared"]
            * self.scale**2
            * (
                1
                if self._prediction_step == 0
                else self.cell["continuation_squared_norm_multiplier"]
            )
        )
        method = self.cell["method"]
        if method in ("dense",):
            y, raw = realize_dense(h, direction, squared_edit_norm_limit)
            cosine = raw["cosine"]
        elif method == "normalized_dense":
            y, raw = realize_global_normalization(h, direction, squared_edit_norm_limit)
            cosine = raw["planned_chord_cosine"]
        else:
            raise ValueError("Unknown intervention.")
        delta = y.double() - h.double()
        before = h.double().norm(dim=-1)
        drift = (y.double().norm(dim=-1) / before.clamp_min(1e-300) - 1).abs()
        squared_edit_norm = delta.square().sum(-1)
        zeros = torch.zeros_like(squared_edit_norm)
        diag = dict(
            realized_squared_edit_norm=squared_edit_norm,
            requested_squared_edit_norm=torch.full_like(squared_edit_norm, squared_edit_norm_limit),
            cosine=cosine,
            norm_relative_drift=drift,
            changed_coordinates=(delta != 0).sum(-1).double(),
            beta=raw.get("beta", torch.ones_like(squared_edit_norm)),
            capacity_limited=raw.get("capacity_limited", zeros).double(),
            numerical_backoff=raw.get("numerical_backoff", zeros).double(),
            no_op=(squared_edit_norm == 0).double(),
            outside_support_squared_edit_norm=zeros,
        )
        if (
            not bool(torch.isfinite(y).all())
            or bool((squared_edit_norm > squared_edit_norm_limit).any())
            or bool((cosine < 0.9).any())
        ):
            raise ValueError("Actual squared edit norm and cosine gate failed.")
        if method in ("normalized_dense",) and bool((drift > 0.001).any()):
            raise ValueError("Actual full norm preservation gate failed.")
        if method in ("dense",) and bool(
            (abs(squared_edit_norm / squared_edit_norm_limit - 1) > 0.01).any()
        ):
            raise ValueError("Additive squared edit norm match failed.")
        records.append(diag)
        names = list(diag)
        values = torch.stack([diag[k] for k in names], -1).cpu().numpy()
        for context, value in zip(self._contexts, values):
            self.prediction_trace.append(
                dict(
                    context,
                    prediction_step=self._prediction_step,
                    **dict(zip(names, map(float, value))),
                )
            )
        self._prediction_step += 1
        changed = hidden.clone()
        changed[:, -1, :] = y
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return hook, records


def install_sparse(runtime, torch):
    from sparse_steering.baselines.dense import realize_dense
    from sparse_steering.method.constrained_normalization import realize_constrained_normalization

    def make_hook(self, direction, target_norm=0.0, *, matched=True, signs=None):
        if self.cell["method"] == "sparsification_global_norm":
            from sparse_steering.method.global_normalization import make_hook as global_hook

            assert matched and signs is None
            return global_hook(self, torch, direction)
        assert (
            matched and signs is None and (self.cell["method"] in ("ours", "sparsification_only"))
        )
        records = []

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states = hidden[:, -1, :]
            assert states.dtype == torch.bfloat16
            squared_edit_norm_limit = (
                self.cell["edit_norm_ratio_squared"]
                * self.scale**2
                * (
                    1
                    if self._prediction_step == 0
                    else self.cell["continuation_squared_norm_multiplier"]
                )
            )
            scaling = self.cell["method"] == "ours"
            if scaling:
                updated, raw = realize_constrained_normalization(
                    states, direction, self.extra_normalization_mask, squared_edit_norm_limit
                )
                squared_edit_norm = (updated.double() - states.double()).square().sum(-1)
                torch.testing.assert_close(
                    squared_edit_norm, raw["realized_squared_edit_norm"], rtol=0, atol=1e-10
                )
                diag = dict(
                    realized_squared_edit_norm=squared_edit_norm,
                    requested_squared_edit_norm=torch.full_like(
                        squared_edit_norm, squared_edit_norm_limit
                    ),
                    cosine=raw["planned_chord_cosine"],
                    norm_relative_drift=raw["norm_relative_drift"],
                    outside_support_squared_edit_norm=raw["outside_support_squared_edit_norm"],
                    beta=raw["beta"],
                    alpha=raw["alpha"],
                    effective_target_squared_edit_norm=raw["effective_target_squared_edit_norm"],
                    physical_squared_edit_norm_limit=raw["physical_squared_edit_norm_limit"],
                    target_relative_error=raw["target_relative_error"],
                    capacity_limited=raw["capacity_limited"].double(),
                    numerical_backoff=raw["numerical_backoff"].double(),
                    no_op=raw["no_op"].double(),
                    changed_coordinates=raw["changed_coordinates"].double(),
                    support_unchanged=raw["support_unchanged"].double(),
                )
                assert not bool((squared_edit_norm > squared_edit_norm_limit).any()) and (
                    not bool((raw["norm_relative_drift"] > 0.001).any())
                )
            else:
                updated, raw = realize_dense(states, direction, squared_edit_norm_limit)
                delta = updated.double() - states.double()
                support = direction != 0
                squared_edit_norm = delta.square().sum(-1)
                aligned = (delta * direction >= 0).all(-1)
                before, after = (states.double().norm(dim=-1), updated.double().norm(dim=-1))
                assert torch.equal(updated[:, ~support], states[:, ~support]) and bool(
                    aligned.all()
                )
                assert bool(torch.isfinite(updated).all()) and (
                    not bool((squared_edit_norm > squared_edit_norm_limit).any())
                )
                assert not bool(
                    (abs(squared_edit_norm / squared_edit_norm_limit - 1) > 0.01).any()
                ) and (not bool((raw["cosine"] < 0.9).any()))
                diag = dict(
                    realized_squared_edit_norm=squared_edit_norm,
                    requested_squared_edit_norm=torch.full_like(
                        squared_edit_norm, squared_edit_norm_limit
                    ),
                    cosine=raw["cosine"],
                    norm_relative_drift=(after / before.clamp_min(1e-300) - 1).abs(),
                    norm_before=before,
                    norm_after=after,
                    changed_coordinates=(delta != 0).sum(-1).double(),
                    outside_support_squared_edit_norm=delta[:, ~support].square().sum(-1),
                    support_unchanged=(updated[:, ~support] == states[:, ~support])
                    .all(-1)
                    .double(),
                    displacement_sign_aligned=aligned.double(),
                    quantization_adjustment_steps=raw["quantization_adjustment_steps"].double(),
                )
            records.append(diag)
            names = list(diag)
            values = torch.stack([diag[k].double() for k in names], -1).cpu().tolist()
            for context, value in zip(self._contexts, values):
                prediction_step = dict(
                    context, prediction_step=self._prediction_step, **dict(zip(names, value))
                )
                if scaling:
                    prediction_step.update(
                        steering_coordinate_count=raw["steering_coordinate_count"],
                        normalization_coordinates=raw["normalization_support_count"],
                    )
                self.prediction_trace.append(prediction_step)
            self._prediction_step += 1
            changed = hidden.clone()
            changed[:, -1, :] = updated
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        return hook, records

    runtime.make_hook = types.MethodType(make_hook, runtime)


def install_coordinate(runtime, torch):
    from sparse_steering.baselines.dense import realize_dense

    def make_hook(self, direction, target_norm=0.0, *, matched=True, signs=None):
        if not matched or signs is not None or self.cell["method"] != "sparsification_only":
            raise ValueError("Coordinate selectors require a positive matched additive update.")
        records = []

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states = hidden[:, -1, :]
            if states.dtype != torch.bfloat16:
                raise ValueError("Expected BF16 states.")
            squared_edit_norm_limit = (
                self.cell["edit_norm_ratio_squared"]
                * self.scale**2
                * (
                    1
                    if self._prediction_step == 0
                    else self.cell["continuation_squared_norm_multiplier"]
                )
            )
            updated, raw = realize_dense(states, direction, squared_edit_norm_limit)
            delta = updated.double() - states.double()
            squared_edit_norm = delta.square().sum(-1)
            support = direction != 0
            unchanged = (updated[:, ~support] == states[:, ~support]).all(-1)
            aligned = (delta * direction >= 0).all(-1)
            if not bool(torch.isfinite(updated).all()) or not bool(unchanged.all() & aligned.all()):
                raise ValueError("Additive update changed sign or zero coordinates.")
            if (
                bool((squared_edit_norm > squared_edit_norm_limit).any())
                or bool((abs(squared_edit_norm / squared_edit_norm_limit - 1) > 0.01).any())
                or bool((raw["cosine"] < 0.9).any())
            ):
                raise ValueError("Original displacement/cosine gates failed.")
            before = states.double().norm(dim=-1)
            zeros = torch.zeros_like(squared_edit_norm)
            diag = dict(
                realized_squared_edit_norm=squared_edit_norm,
                requested_squared_edit_norm=torch.full_like(
                    squared_edit_norm, squared_edit_norm_limit
                ),
                cosine=raw["cosine"],
                norm_relative_drift=(
                    updated.double().norm(dim=-1) / before.clamp_min(1e-300) - 1
                ).abs(),
                changed_coordinates=(delta != 0).sum(-1).double(),
                outside_support_squared_edit_norm=delta[:, ~support].square().sum(-1),
                support_unchanged=unchanged.double(),
                displacement_sign_aligned=aligned.double(),
                quantization_adjustment_steps=raw["quantization_adjustment_steps"].double(),
                beta=torch.ones_like(squared_edit_norm),
                capacity_limited=zeros,
                numerical_backoff=zeros,
                no_op=(squared_edit_norm == 0).double(),
            )
            if hasattr(self, "component_directions"):
                diag.update(
                    projection_truthfulqa=(delta * self.component_directions[0]).sum(-1),
                    projection_halueval=(delta * self.component_directions[1]).sum(-1),
                )
            records.append(diag)
            names = list(diag)
            values = torch.stack([diag[k].double() for k in names], -1).cpu().tolist()
            for context, value in zip(self._contexts, values):
                self.prediction_trace.append(
                    dict(context, prediction_step=self._prediction_step, **dict(zip(names, value)))
                )
            self._prediction_step += 1
            changed = hidden.clone()
            changed[:, -1, :] = updated
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        return hook, records

    runtime.make_hook = types.MethodType(make_hook, runtime)
