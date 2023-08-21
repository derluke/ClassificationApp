import base64
import os
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from PIL import Image
from pydantic import BaseModel  # pylint: disable=no-name-in-module
import datarobot as dr
import altair as alt
import deployment_patch

load_dotenv()

im = Image.open("./image/datarobot.jpg")

st.set_page_config(
    # layout="wide",
    page_icon=im,
    page_title="Customer Classification App",  # edit this for your usecase
    initial_sidebar_state="collapsed",
)
st.title("Customer Classification App")


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


header_html = f"""
<style>
    #content {{
        position: relative;
    }}
    #content img {{
        position: absolute;
        top: -50px;
        right: -30px;
    }}
</style>
<div id='content'>
    <img src='data:image/png;base64,{img_to_bytes('image/Robot-icon-blue-eyes_transparent.png')}'
    class='img-fluid' width='50'>
</div>
"""

st.markdown(
    header_html,
    unsafe_allow_html=True,
)


def color_text(text, segments):
    colored_text = ""
    last_end = 0

    for segment in segments:
        ngram = segment["ngrams"][0]
        start, end = ngram["startingIndex"], ngram["endingIndex"]

        # Append any uncolored text
        colored_text += text[last_end:start]

        # Determine the color based on qualitativeStrength
        base_color = {
            True: (237, 29, 14),
            False: (19, 7, 240),  # bright blue
        }.get(
            segment["strength"] > 0, (255, 255, 255)
        )  # default to white

        # Determine the alpha value based on strength
        alpha = segment["strength"] / 2

        # Append colored text
        colored_text += f"<span style='background-color: rgba{base_color + (alpha,)}; color: white; font-size: 1.5em'>{text[start:end+1]}</span>"

        last_end = end + 1

    # Append any remaining uncolored text
    colored_text += text[last_end:]

    return colored_text


client = dr.Client(
    token=os.environ["DATAROBOT_API_TOKEN"], endpoint=os.environ["DATAROBOT_ENDPOINT"]
)
deployment = dr.Deployment.get("64e3e04b0f985aa65ea64ca4")


def translate_to_english(text):
    text_en = GoogleTranslator(source="auto", target="en").translate(text)
    return text_en


def setup_llm():
    OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
    OPENAI_ORGANIZATION = os.environ["OPENAI_ORGANIZATION"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_TYPE = os.environ["OPENAI_API_TYPE"]
    OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
    OPENAI_DEPLOYMENT_NAME = os.environ["OPENAI_DEPLOYMENT_NAME"]

    llm = AzureChatOpenAI(
        deployment_name=OPENAI_DEPLOYMENT_NAME,
        openai_api_type=OPENAI_API_TYPE,
        openai_api_base=OPENAI_API_BASE,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=OPENAI_API_KEY,
        openai_organization=OPENAI_ORGANIZATION,
        model_name=OPENAI_DEPLOYMENT_NAME,
        temperature=0.2,
        verbose=True,
        max_retries=0,
        request_timeout=20,
    )
    return llm


raw_industry_classification_df = pd.read_csv("data/GICSMap2023.csv")


def prep_hierarchy_table(raw_df):
    raw_df = raw_df.apply(
        lambda col: col.str.replace("(New Name)", "", regex=False)
        if col.dtype == "object"
        else col
    )
    raw_df = raw_df.apply(
        lambda col: col.str.replace("(New name)", "", regex=False)
        if col.dtype == "object"
        else col
    )
    raw_df = raw_df.apply(
        lambda col: col.str.replace("(New Code)", "", regex=False)
        if col.dtype == "object"
        else col
    )
    raw_df = raw_df.apply(
        lambda col: col.str.replace("(Definition Update)", "", regex=False)
        if col.dtype == "object"
        else col
    )
    raw_df = raw_df.apply(
        lambda col: col.str.replace("(New)", "", regex=False)
        if col.dtype == "object"
        else col
    )
    raw_df = raw_df[~raw_df.SubIndustry.str.contains("(Discontinued)", regex=False)]
    raw_df = raw_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # replace multiple whitespace with single ' ':
    raw_df = raw_df.apply(
        lambda col: col.str.replace(r"\s+", " ", regex=True)
        if col.dtype == "object"
        else col
    )

    # replace anything that's inside '(...)' with '':
    raw_df = raw_df.apply(
        lambda col: col.str.replace(r"\(.*\)", "", regex=True)
        if col.dtype == "object"
        else col
    )

    raw_df = raw_df.replace(r"^\s*$", np.nan, regex=True)

    compact_industries = (
        raw_df[
            [
                "SectorId",
                "Sector",
                "IndustryGroupId",
                "IndustryGroup",
                "IndustryId",
                "Industry",
                "SubIndustryId",
                "SubIndustry",
                "description",
            ]
        ]
        .ffill()
        .drop_duplicates()
    )

    return compact_industries


def create_enum(name, values):
    """
    Create an enum from a dictionary of values in order to programmatically create a pydantic model

    Args:
        name (str): name of the enum
        values (dict): dictionary of values

    Returns:
        Enum: enum object
    """
    return Enum(name, values)


industry_classification_df = prep_hierarchy_table(raw_industry_classification_df)


fig2 = px.sunburst(
    industry_classification_df,
    path=["Sector", "IndustryGroup", "Industry", "SubIndustry"],
    color="Sector",
)
fig2.update_layout(title_text="GICS Industry hierarchy", font_size=10)

sectors = set(industry_classification_df["Sector"].drop_duplicates())
industry_groups = set(industry_classification_df["IndustryGroup"].drop_duplicates())
industries = set(industry_classification_df["Industry"].drop_duplicates())
sub_industries = set(industry_classification_df["SubIndustry"].drop_duplicates())


SectorEnum = create_enum("Sector", {sector.upper(): sector for sector in sectors})
IndustryGroupEnum = create_enum(
    "IndustryGroup",
    {industry_group.upper(): industry_group for industry_group in industry_groups},
)
IndustryEnum = create_enum(
    "Industry", {industry.upper(): industry for industry in industries}
)
SubIndustryEnum = create_enum(
    "SubIndustry",
    {sub_industry.upper(): sub_industry for sub_industry in sub_industries},
)


class SectorModel(BaseModel):
    sector: SectorEnum


class IndustryGroupModel(BaseModel):
    industry_group: IndustryGroupEnum


class IndustryModel(BaseModel):
    industry: IndustryEnum


class SubIndustryModel(BaseModel):
    sub_industry: SubIndustryEnum


sector_parser = PydanticOutputParser(pydantic_object=SectorModel)
industry_group_parser = PydanticOutputParser(pydantic_object=IndustryGroupModel)
industry_parser = PydanticOutputParser(pydantic_object=IndustryModel)
sub_industry_parser = PydanticOutputParser(pydantic_object=SubIndustryModel)


sector_classification_prompt = PromptTemplate(
    template=(
        "This is a classification task. Please classify the following text into its most likely industry: "
        "{text}\n\n{format_instructions}"
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": sector_parser.get_format_instructions()},
)

industry_group_classification_prompt = PromptTemplate(
    template=(
        "This is a classification task. Please classify the following text into its most likely industry: "
        "{text}\n\n{format_instructions}"
    ),
    input_variables=["text"],
    partial_variables={
        "format_instructions": industry_group_parser.get_format_instructions()
    },
)

industry_classification_prompt = PromptTemplate(
    template=(
        "This is a classification task. Please classify the following text into its most likely industry: "
        "{text}\n\n{format_instructions}"
    ),
    input_variables=["text"],
    partial_variables={
        "format_instructions": industry_parser.get_format_instructions()
    },
)

sub_industry_classification_prompt = PromptTemplate(
    template=(
        "This is a classification task. Please classify the following text into its most likely industry: "
        "{text}\n\n{format_instructions}"
    ),
    input_variables=["text"],
    partial_variables={
        "format_instructions": sub_industry_parser.get_format_instructions()
    },
)

llm = setup_llm()


def get_classification(text):
    """
    Get the industry classification for text.
    This is a simple algorithm that first tries to classify the text into a industry, then a sub industry.
    If it fails, it retries three times. This works since the model is stochastic due to the temperature parameter.

    Args:
        text (str): text to classify

    Returns:
        None
    """
    for _ in range(3):
        output = llm.predict(industry_classification_prompt.format(text=text))
        try:
            result = industry_parser.parse(output)
        except:
            st.warning(f"retrying - found {output}")
            output = llm.predict(sub_industry_classification_prompt.format(text=text))
            try:
                result = sub_industry_parser.parse(output)
            except:
                st.warning(f"retrying - found {output}")
                continue
            else:
                break
        else:
            break
    else:
        st.error("failed to get valid output")

    st.success(result)


with st.expander("Industry Classification Table"):
    st.write(industry_classification_df)

with st.expander("Industry Classification Chart"):
    st.plotly_chart(fig2, use_container_width=True)

text = st.text_area("Text to classify", height=200)
if st.button("Classify"):
    get_classification(text)

    preds = dr.Deployment.get("64e3e04b0f985aa65ea64ca4").predict(
        pd.DataFrame.from_records([{"CR_ACTIVITY": text}]),
        max_explanations=1,
        textExplanations=True,
    )
    predictions_sorted = pd.DataFrame.from_records(
        preds["predictionValues"].values[0]
    ).sort_values("value", ascending=False)
    prob_to_plot = pd.concat(
        [
            predictions_sorted.iloc[:5],
            pd.DataFrame.from_records(
                [
                    {
                        "label": "All remaining",
                        "value": predictions_sorted.iloc[5:].value.sum(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).reset_index(drop=True)
    st.markdown("### DataRobot Predictions")
    # st.bar_chart(prob_to_plot, x="label", y="value", use_container_width=True)
    st.altair_chart(
        alt.Chart(prob_to_plot)
        .mark_bar()
        .encode(
            x=alt.X("label", sort=None),
            y=alt.Y("value", title="Probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=["label", "value"],
        ),
        use_container_width=True,
    )

    segments = preds["Explanation 1 perNgramTextExplanations"].values[0]
    colored_output = color_text(text, segments)
    predicted_value = predictions_sorted.iloc[0].label
    st.markdown(f'### Explanation for predicted value: "{predicted_value}"')
    st.markdown(colored_output, unsafe_allow_html=True)
    st.markdown("### Translation")
    st.write(translate_to_english(text))
