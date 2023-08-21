import datarobot as dr
import pandas as pd


def get_multiseries_id_columns(self):
    return (
        dr.DatetimePartitioning.get(self.model["project_id"])
        .multiseries_id_columns[0]
        .replace(" (actual)", "")
    )


def get_date_format(self):
    return dr.DatetimePartitioning.get(self.model["project_id"]).date_format


def get_datetime_partition_column(self):
    return dr.DatetimePartitioning.get(
        self.model["project_id"]
    ).datetime_partition_column.replace(" (actual)", "")


def predict(
    self,
    data,
    forecast_point=None,
    predictions_start_date=None,
    predictions_end_date=None,
    max_explanations=None,
    threshold_high=None,
    threshold_low=None,
    textExplanations=None,
):
    url = self.default_prediction_server[
        "url"
    ] + "/predApi/v1.0/deployments/{deployment_id}/predictions".format(
        deployment_id=self.id
    )
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": self._client.headers["Authorization"],
        "DataRobot-Key": self.default_prediction_server["datarobot-key"],
    }
    params = {
        "forecastPoint": forecast_point,
        "predictionsStartDate": predictions_start_date,
        "predictionsEndDate": predictions_end_date,
        # If explanations are required, uncomment the line below
        # 'maxExplanations': 3,
        # 'thresholdHigh': 0.5,
        # 'thresholdLow': 0.15,
        # Uncomment this for Prediction Warnings, if enabled for your deployment.
        # 'predictionWarningEnabled': 'true',
    }
    if max_explanations:
        params["maxExplanations"] = max_explanations
        params["thresholdHigh"] = threshold_high
        params["thresholdLow"] = threshold_low

    if textExplanations:
        params["maxNgramExplanations"] = "all"

    project = dr.Project.get(self.model["project_id"])
    if project.use_time_series:
        date_column = self.get_datetime_partition_column()
        date_format = self.get_date_format()

        if pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = data[date_column].dt.strftime(date_format)
        else:
            data[date_column] = pd.to_datetime(data[date_column]).dt.strftime(
                date_format
            )
    predictions_response = dr.rest.requests.post(
        url, data=data.to_json(orient="records"), headers=headers, params=params
    )
    preds = pd.DataFrame.from_records(predictions_response.json()["data"])
    if project.use_time_series:
        preds["forecastPoint"] = pd.to_datetime(preds["forecastPoint"]).dt.tz_convert(
            None
        )
        preds["timestamp"] = pd.to_datetime(preds["timestamp"]).dt.tz_convert(None)
        try:
            multiseries_id_columns = self.get_multiseries_id_columns()
            preds = preds.rename(columns={"seriesId": multiseries_id_columns})
        except:
            pass
    if project.positive_class:
        prediction_values_temp = pd.DataFrame.from_records(
            preds.predictionValues.values
        )
        for value in prediction_values_temp.columns:
            prediction_values_df = pd.json_normalize(prediction_values_temp[value])
            if (prediction_values_df["label"] == project.positive_class).all():
                positive_preds = prediction_values_df["value"]
                positive_preds.name = "positiveClassPrediction"
                break
        preds = pd.concat(
            [
                preds.drop(columns=["predictionValues"]).reset_index(drop=True),
                positive_preds.reset_index(drop=True),
            ],
            axis=1,
        )
    if max_explanations:
        explanation_temp = pd.DataFrame.from_records(
            preds.predictionExplanations.values
        )
        explanation_dfs = []
        for explanation in explanation_temp.columns:
            explanation_df = pd.json_normalize(explanation_temp[explanation])
            explanation_df.columns = [
                f"Explanation {explanation+1} {c}" for c in explanation_df.columns
            ]
            explanation_dfs.append(explanation_df)
        explanation_dfs = pd.concat(explanation_dfs, axis=1)
        preds = pd.concat(
            [
                preds.drop(columns=["predictionExplanations"]).reset_index(drop=True),
                explanation_dfs.reset_index(drop=True),
            ],
            axis=1,
        )
    return preds


dr.Deployment.get_multiseries_id_columns = get_multiseries_id_columns
dr.Deployment.get_date_format = get_date_format
dr.Deployment.get_datetime_partition_column = get_datetime_partition_column
dr.Deployment.predict = predict
