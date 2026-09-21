from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np

from ..dataset import (
    ABCRole,
    AdditionalMatchingRole,
    AdditionalTargetRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    InfoRole,
    TargetRole,
    TreatmentRole,
)
from ..extensions.scipy_stats import NormCDF
from ..utils.enums import ExperimentDataEnum
from ..utils.errors import NoneArgumentError
from .abstract import GroupOperator


class SMD(GroupOperator):
    """
    Standardized Mean Difference between control and test groups.

    By default:
        - TreatmentRole is used to split observations into control/test groups;
        - FeatureRole columns are used as variables for SMD calculation;
        - control group is encoded as 0;
        - test group is encoded as 1.

    For every feature:

        SMD = (mean_test - mean_control) / pooled_std
    """

    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        control_value: Any = 0,
        test_value: Any = 1,
        key: Any = "",
    ):
        super().__init__(
            grouping_role=grouping_role or TreatmentRole(),
            target_roles=target_roles or FeatureRole(),
            key=key,
        )

        self.control_value = control_value
        self.test_value = test_value

    def _get_fields(
        self,
        data: ExperimentData,
    ) -> tuple[list[str], list[str]]:
        """
        Get treatment field and features.

        GroupOperator._get_fields() assumes two target columns,
        which is not appropriate for SMD: SMD should support
        an arbitrary number of features.
        """

        group_field = data.field_search(self.grouping_role)

        target_fields = data.field_search(
            self.target_roles,
            search_types=self.search_types,
        )

        return group_field, target_fields

    def calculate(
        self,
        data: ExperimentData,
    ) -> dict[str, float]:
        """Calculate SMD for each selected feature."""
        group_field, target_fields = self._get_fields(data=data)
        if not group_field:
            raise ValueError(
                "SMD requires a grouping column "
                "(TreatmentRole by default)."
            )
        if not target_fields:
            raise ValueError(
                "SMD requires at least one feature "
                "(FeatureRole by default)."
            )
        return self.calc(
            data=data.ds,
            group_field=group_field,
            target_fields=target_fields,
            control_value=self.control_value,
            test_value=self.test_value,
        )
    def execute(
        self,
        data: ExperimentData,
    ) -> ExperimentData:
        """Calculate SMD and store the result in ExperimentData."""
        _, target_fields = self._get_fields(data=data)
        if not target_fields and data.ds.tmp_roles:
            return data
        self.key = str(
            target_fields[0]
            if len(target_fields) == 1
            else (target_fields or "")
        )
        compare_result = self.calculate(data)
        return self._set_value(
            data,
            compare_result,
        )


    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: list[str] | None = None,
        control_value: Any = 0,
        test_value: Any = 1,
        **kwargs,
    ) -> dict[str, float]:
        """
        Calculate one SMD for each feature.

        grouping_data contains two datasets:
            control group
            test group
        """

        if not target_fields:
            raise ValueError(
                "SMD requires at least one feature."
            )

        if len(grouping_data) != 2:
            raise ValueError(
                "SMD requires exactly 2 groups "
                f"(control and test), got {len(grouping_data)}."
            )

        groups = {}

        for group_key, group_data in grouping_data:
            # Dataset.groupby() may return keys as one-element tuples.
            if (
                isinstance(group_key, (tuple, list))
                and len(group_key) == 1
            ):
                group_key = group_key[0]

            groups[group_key] = group_data

        if control_value not in groups:
            raise ValueError(
                f"Control group {control_value!r} was not found. "
                f"Available groups: {list(groups.keys())}"
            )

        if test_value not in groups:
            raise ValueError(
                f"Test group {test_value!r} was not found. "
                f"Available groups: {list(groups.keys())}"
            )

        control_data = groups[control_value]
        test_data = groups[test_value]

        result = {}

        for field in target_fields:
            result[field] = cls._inner_function(
                data=control_data[field],
                test_data=test_data[field],
                **kwargs,
            )

        return result

    @staticmethod
    def _to_numpy(
        data: Dataset,
    ) -> np.ndarray:
        """
        Convert a single-column Dataset to a 1D float NumPy array.
        """

        values = data.data

        # pandas DataFrame / Series
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()

        values = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)

        # pandas mean()/var() normally ignore NaN,
        # so preserve equivalent behaviour here.
        values = values[~np.isnan(values)]

        return values

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        **kwargs,
    ) -> float:
        """
        Calculate signed Standardized Mean Difference:

            (mean_test - mean_control) / pooled_std

        Sample variances are calculated with ddof=1.
        """

        test_data = cls._check_test_data(
            test_data=test_data
        )

        control = cls._to_numpy(data)
        test = cls._to_numpy(test_data)

        n_control = control.size
        n_test = test.size

        if n_control < 2 or n_test < 2:
            raise ValueError(
                "SMD requires at least 2 observations "
                "in both control and test groups."
            )

        mean_control = np.mean(control)
        mean_test = np.mean(test)

        var_control = np.var(
            control,
            ddof=1,
        )
        var_test = np.var(
            test,
            ddof=1,
        )

        pooled_variance = (
            (n_control - 1) * var_control
            + (n_test - 1) * var_test
        ) / (
            n_control + n_test - 2
        )

        pooled_std = np.sqrt(
            pooled_variance
        )

        mean_difference = (
            mean_test - mean_control
        )

        # Explicit NumPy division gives mathematically natural
        # behaviour for zero variance:
        #   non-zero difference / 0 -> +/-inf
        #   0 / 0                   -> nan
        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            smd = np.divide(
                mean_difference,
                pooled_std,
            )

        return float(smd)


class MatchingMetrics(GroupOperator):
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        metric: Literal["auto", "atc", "att", "ate"] | None = None,
        n_neighbors: int = 1,
        key: Any = "",
    ):
        self.metric = metric or "auto"
        self.n_neighbors = n_neighbors
        self.__scaled_counts = {}
        target_roles = target_roles or TargetRole()
        super().__init__(
            grouping_role=grouping_role,
            target_roles=(
                target_roles if isinstance(target_roles, list) else [target_roles]
            ),
            key=key,
        )

    def _calc_scaled_counts(self, matches, indexes, group):
        matches_counts = Dataset({})
        matches_counts = matches_counts.add_column(
            indexes.index, {"indexes": InfoRole()}
        )
        matches_counts = matches_counts.add_column([0], {"count": InfoRole(float)})
        for col in matches.columns:
            v_counts = matches[col].value_counts()
            matches_counts = matches_counts.merge(
                v_counts,
                how="left",
                left_on="indexes",
                right_on=col,
                suffixes=(("", col)),
            ).drop(columns=col)
        matches_counts.index = indexes.index
        matches_counts = matches_counts.drop(columns="indexes").fillna(0)
        for col in matches_counts.columns:
            if col != "count":
                matches_counts["count"] += matches_counts[col]
        self.__scaled_counts[group] = matches_counts["count"] / self.n_neighbors

    @staticmethod
    def _calc_vars(value):
        var = 0 if value[value.columns[0]].isna().sum() > 0 else value.var()
        return value * 0 + var

    @staticmethod
    def _calc_se(var_c, var_t, scaled_counts, group=None):
        n_c, n_t = len(var_c), len(var_t)
        if group is not None:
            groups = list(scaled_counts.keys())
            groups.remove(group)
            group_other = groups[0]
            weights_c = scaled_counts[group_other] * 0 + 1
            weights_t = scaled_counts[group] * n_t / n_c
        else:
            n = n_c + n_t
            weights_c = (n_c / n) * (scaled_counts["test"] + 1)
            weights_t = (n_t / n) * (scaled_counts["control"] + 1)

        return np.sqrt(
            (weights_t**2 * var_t).sum() / n_t**2
            + (weights_c**2 * var_c).sum() / n_c**2
        )

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_fields: list[str] | None = None,
        **kwargs,
    ) -> Any:
        if target_fields is None or test_data is None:
            raise NoneArgumentError(
                ["target_fields", "test_data"], "att, atc, ate estimation"
            )
        metric = kwargs.get("metric", "ate")
        scaled_counts = kwargs.get("scaled_counts")
        itt = test_data[target_fields[0]] - test_data[target_fields[1]]
        itc = data[target_fields[1]] - data[target_fields[0]]
        bias = kwargs.get("bias", {})
        if bias and len(bias) > 0:
            if metric in ["atc", "ate"]:
                itc -= Dataset.from_dict(
                    {"test": bias["control"]}, roles={}, index=itc.index
                )
            if metric in ["att", "ate"]:
                itt += Dataset.from_dict(
                    {"control": bias["test"]}, roles={}, index=itt.index
                )
        var_t = cls._calc_vars(itc)
        var_c = cls._calc_vars(itt)
        itt_se = cls._calc_se(var_c, var_t, scaled_counts, "control")
        itc_se = cls._calc_se(var_t, var_c, scaled_counts, "test")
        itt = itt.mean()
        itc = itc.mean()
        p_val_itt = (
            NormCDF()
            .calc(
                Dataset.from_dict(
                    {"value": [itt / itt_se]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
        p_val_itc = (
            NormCDF()
            .calc(
                Dataset.from_dict(
                    {"value": [itc / itc_se]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
        if metric == "atc":
            return {
                "ATC": [
                    itc,
                    itc_se,
                    p_val_itc,
                    itc - 1.96 * itc_se,
                    itc + 1.96 * itc_se,
                ]
            }
        if metric == "att":
            return {
                "ATT": [
                    itt,
                    itt_se,
                    p_val_itt,
                    itt - 1.96 * itt_se,
                    itt + 1.96 * itt_se,
                ]
            }
        len_control, len_test = len(data), len(test_data)
        ate = (itt * len_test + itc * len_control) / (len_test + len_control)
        ate_se = cls._calc_se(var_c, var_t, scaled_counts)
        p_val_ate = (
            NormCDF()
            .calc(
                Dataset.from_dict(
                    {"value": [ate / ate_se]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
        return {
            "ATT": [itt, itt_se, p_val_itt, itt - 1.96 * itt_se, itt + 1.96 * itt_se],
            "ATC": [itc, itc_se, p_val_itc, itc - 1.96 * itc_se, itc + 1.96 * itc_se],
            "ATE": [ate, ate_se, p_val_ate, ate - 1.96 * ate_se, ate + 1.96 * ate_se],
        }

    @classmethod
    def _execute_inner_function(
        cls, grouping_data, target_fields: list[str] | None = None, **kwargs
    ) -> dict:
        metric = kwargs.get("metric", "ate")
        if target_fields is None or len(target_fields) != 2:
            raise ValueError(
                f"This operator works with 2 targets, but got {len(target_fields) if target_fields else None}"
            )
        return cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            metric=metric,
            bias=kwargs.get("bias_estimation", None),
            scaled_counts=kwargs.get("scaled_counts"),
        )

    def _prepare_new_target(
        self,
        data: ExperimentData,
        t_data: Dataset,
        group_field: str,
    ) -> Dataset:
        new_target = data.ds.search_columns(TargetRole())[0]
        indexes, matched_data = Bias.prepare_data(data, t_data)
        matched_data = matched_data[new_target + "_matched"]
        grouped_data = data.ds.groupby(group_field)
        control_indexes = indexes.loc[grouped_data[0][1].index, :]
        test_indexes = indexes.loc[grouped_data[1][1].index, :]
        self._calc_scaled_counts(control_indexes, test_indexes, "test")
        self._calc_scaled_counts(test_indexes, control_indexes, "control")

        return matched_data

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data=data)
        bias = (
            data.variables[data.get_one_id(Bias, ExperimentDataEnum.variables)]
            if len(
                data.get_ids(Bias, ExperimentDataEnum.variables)["Bias"]["variables"]
            )
            > 0
            else None
        )
        t_data = deepcopy(data.ds)
        if len(target_fields) != 2:
            matched_data = self._prepare_new_target(data, t_data, group_field)
            target_fields += [matched_data.search_columns(TargetRole())[0]]
            data.set_value(
                ExperimentDataEnum.additional_fields,
                self.id,
                matched_data,
                role=AdditionalTargetRole(),
            )
            t_data = t_data.add_column(
                matched_data.reindex(t_data.index),
                role={target_fields[1]: TargetRole()},
            )
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            return data

        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            metric=self.metric,
            bias_estimation=bias,
            scaled_counts=self.__scaled_counts,
        )
        return self._set_value(data, compare_result)


class Bias(GroupOperator):
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: list[ABCRole] | None = None,
        key: Any = "",
    ):
        super().__init__(
            grouping_role=grouping_role, target_roles=target_roles, key=key
        )

    @staticmethod
    def calc_coefficients(X: Dataset, Y: Dataset) -> list[float]:
        X_l = Dataset.create_empty(roles={"temp": InfoRole()}, index=X.index).fillna(1)
        X = X_l.append(X, axis=1).data.values
        return np.linalg.lstsq(X, Y.data.values, rcond=-1)[0][1:]

    @staticmethod
    def calc_bias(
        X: Dataset, X_matched: Dataset, coefficients: list[float]
    ) -> list[float]:
        return [
            (j - i).dot(coefficients)[0]
            for i, j in zip(X.data.values, X_matched.data.values)
        ]

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_fields: list[str] | None = None,
        features_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        if target_fields is None or features_fields is None or test_data is None:
            raise NoneArgumentError(
                ["target_fields", "features_fields", "test_data"], "bias_estimation"
            )
        if data[target_fields[1]].isna().sum() > 0:
            return {
                "test": cls.calc_bias(
                    test_data[features_fields[: len(features_fields) // 2]],
                    test_data[features_fields[len(features_fields) // 2 :]],
                    cls.calc_coefficients(
                        test_data[features_fields[len(features_fields) // 2 :]],
                        test_data[target_fields[1]],
                    ),
                )
            }
        if test_data[target_fields[1]].isna().sum() > 0:
            return {
                "control": cls.calc_bias(
                    data[features_fields[: len(features_fields) // 2]],
                    data[features_fields[len(features_fields) // 2 :]],
                    cls.calc_coefficients(
                        data[features_fields[len(features_fields) // 2 :]],
                        data[target_fields[1]],
                    ),
                )
            }
        return {
            "test": cls.calc_bias(
                test_data[features_fields[: len(features_fields) // 2]],
                test_data[features_fields[len(features_fields) // 2 :]],
                cls.calc_coefficients(
                    test_data[features_fields[len(features_fields) // 2 :]],
                    test_data[target_fields[1]],
                ),
            ),
            "control": cls.calc_bias(
                data[features_fields[: len(features_fields) // 2]],
                data[features_fields[len(features_fields) // 2 :]],
                cls.calc_coefficients(
                    data[features_fields[len(features_fields) // 2 :]],
                    data[target_fields[1]],
                ),
            ),
        }

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: list[str] | None = None,
        features_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        return cls._inner_function(
            grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            features_fields=features_fields,
            **kwargs,
        )

    @staticmethod
    def prepare_data(data: ExperimentData, t_data: Dataset) -> Dataset:
        indexes = data.field_search(AdditionalMatchingRole())
        if len(indexes) == 0:
            raise ValueError("No indexes were found")
        indexes = data.additional_fields[indexes]
        indexes.index = t_data.index
        filtered_field = indexes.drop(
            indexes[indexes[indexes.columns[0]] == -1], axis=0
        )
        matched_data = Dataset({})
        matched_data.index = filtered_field.index
        numeric_cols = t_data.search_columns(
            [FeatureRole(), TargetRole()], search_types=[int, float]
        )
        for d_col in numeric_cols:
            matched_data_col = Dataset({})
            matched_data_col.index = filtered_field.index
            for i, i_col in enumerate(indexes.columns):
                index_matched_data = data.ds.loc[
                    list(filtered_field[i_col].get_values(column=i_col))
                ][d_col].rename(
                    {d_col: d_col + f"_matched_{i}" for _ in data.ds.columns}
                )
                matched_data_col = matched_data_col.add_column(index_matched_data)
            default_value = [t_data.roles[d_col].data_type(0)]
            matched_data_col = matched_data_col.add_column(
                default_value * matched_data_col.shape[0],
                {d_col + "_matched": t_data.roles[d_col]},
            )
            for col in matched_data_col.columns:
                if col != d_col + "_matched":
                    matched_data_col[d_col + "_matched"] += matched_data_col[col]
            matched_data = matched_data.add_column(
                default_value * matched_data_col.shape[0],
                {d_col + "_matched": t_data.roles[d_col]},
            )
            matched_data[d_col + "_matched"] = matched_data_col[d_col + "_matched"] / (
                matched_data_col.shape[1] - 1
            )

        return indexes, matched_data

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data)
        t_data = deepcopy(data.ds)
        if len(target_fields) < 2:
            _, matched_data = self.prepare_data(data, t_data)
            target_fields += [matched_data.search_columns(TargetRole())[0]]
            t_data = t_data.append(matched_data.reindex(t_data.index), axis=1)
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            return data

        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            features_fields=t_data.search_columns(
                FeatureRole(), search_types=[int, float]
            ),
        )
        return self._set_value(data, compare_result)
