import asyncio
import glob
import os

import numpy as np
import pandas as pd

import streamlit as st
# from st_aggrid import AgGrid
# from st_aggrid.grid_options_builder import GridOptionsBuilder
# from st_aggrid.shared import GridUpdateMode



# @st.cache
def get_csvdf(rulefolder):
    files2 = glob.glob(rulefolder + "**/*.csv", recursive=True)
    dflist = []
    for filepath in files2:
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        newdf = rule2df(filename, filepath)  # [['监管要求', '结构', '条款']]
        dflist.append(newdf)
    if not dflist:
        st.write("No csv files found in " + rulefolder)
        alldf = pd.DataFrame()
    else:
        alldf = pd.concat(dflist, axis=0)
    return alldf


def rule2df(filename, filepath):
    docdf = pd.read_csv(filepath)
    docdf["监管要求"] = filename
    return docdf


# split string by space into words, add brackets before and after words, combine into text
def split_words(text):
    words = text.split()
    words = ["(?=.*" + word + ")" for word in words]
    new = "".join(words)
    return new


# get section list from df
def get_section_list(searchresult, make_choice):
    """
    get section list from df

    args: searchresult, make_choice
    return: section_list
    """
    df = searchresult[(searchresult["监管要求"].isin(make_choice))]
    conls = df["结构"].drop_duplicates().tolist()
    unils = []
    for con in conls:
        itemls = con.split("/")
        for item in itemls[:2]:
            unils.append(item)
    # drop duplicates and keep list order
    section_list = list(dict.fromkeys(unils))
    return section_list


# get folder name list from path
def get_folder_list(path):
    folder_list = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]
    return folder_list


# def df2aggrid(df):
#     gb = GridOptionsBuilder.from_dataframe(df)
#     gb.configure_pagination()
#     gb.configure_side_bar()
#     # gb.configure_auto_height()
#     gb.configure_default_column(
#         genablePivot=True, enableValue=True, enableRowGroup=True, editable=True
#     )
#     gb.configure_selection(selection_mode="single", use_checkbox=True)
#     gridOptions = gb.build()
#     ag_grid = AgGrid(
#         df,
#         theme="material",
#         #  height=800,
#         fit_columns_on_grid_load=True,  # fit columns to grid width
#         gridOptions=gridOptions,  # grid options
#         #  key='select_grid', # key is used to identify the grid
#         update_mode=GridUpdateMode.SELECTION_CHANGED,
#         # data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
#         #  update_mode=GridUpdateMode.NO_UPDATE,
#         enable_enterprise_modules=True,
#     )
#     return ag_grid
