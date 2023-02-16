import ast
import io
import os

import pandas as pd
import streamlit as st

from gptfuc import build_index, gpt_answer
from upload import (

    get_uploadfiles,
    remove_uploadfiles,
    save_uploadedfile,
    savedf,
)
from utils import (
    df2aggrid,
    get_folder_list,
)

auditfolder="audits"

def main():

    # st.subheader("制度匹配分析")
    menu = ["文件上传", "文件选择", "文件问答"]
    choice = st.sidebar.selectbox("选择", menu)

    if choice == "文件上传":
        st.subheader("文件上传")
        # choose input method of manual or upload file
        input_method = st.sidebar.radio("文件上传方式", ("项目文件", "上传底稿"))

        if input_method == "项目文件":

            uploaded_file_ls = st.file_uploader(
                "选择新文件上传",
                type=["docx", "pdf", "txt"],
                accept_multiple_files=True,
                help="选择文件上传",
            )
            for uploaded_file in uploaded_file_ls:
                if uploaded_file is not None:
                        
                    # Check File Type
                    if (
                        (
                            uploaded_file.type
                            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        | (uploaded_file.type == "application/pdf")
                        | (uploaded_file.type == "text/plain")
                    ):
                        save_uploadedfile(uploaded_file)            
 
        elif input_method == "上传底稿":
            uploaded_file_ls = st.file_uploader(
                "选择新文件上传",
                type=["docx", "pdf", "txt", "xlsx"],
                accept_multiple_files=True,
                help="选择文件上传",
            )

            for uploaded_file in uploaded_file_ls:
                if uploaded_file is not None:

                    # Check File Type
                    if (
                        (
                            uploaded_file.type
                            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        | (uploaded_file.type == "application/pdf")
                        | (uploaded_file.type == "text/plain")
                    ):
                        save_uploadedfile(uploaded_file)

                        # if upload file is xlsx
                    elif (
                        uploaded_file.type
                        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ):
                        # get sheet names list from excel file
                        xls = pd.ExcelFile(uploaded_file)
                        sheets = xls.sheet_names
                        # choose sheet name and click button
                        sheet_name = st.selectbox("选择表单", sheets)

                        # choose header row
                        header_row = st.number_input(
                            "选择表头行",
                            min_value=0,
                            max_value=10,
                            value=0,
                            key="header_row",
                        )
                        df = pd.read_excel(
                            uploaded_file, header=header_row, sheet_name=sheet_name
                        )
                        # filllna
                        df = df.fillna("")
                        # display the first five rows
                        st.write(df.astype(str))

                        # get df columns
                        cols = df.columns
                        # choose proc_text and audit_text column
                        proc_col = st.sidebar.selectbox("选择文本列", cols)

                        # get proc_text and audit_text list
                        proc_list = df[proc_col].tolist()

                        # get proc_list and audit_list length
                        proc_len = len(proc_list)

                        # if proc_list or audit_list is empty or not equal
                        if proc_len == 0:
                            st.error("文本列为空，请重新选择")
                            return
                        else:
                            # choose start and end index
                            start_idx = st.sidebar.number_input(
                                "选择开始索引", min_value=0, max_value=proc_len - 1, value=0
                            )
                            end_idx = st.sidebar.number_input(
                                "选择结束索引",
                                min_value=start_idx,
                                max_value=proc_len - 1,
                                value=proc_len - 1,
                            )
                            # get proc_list and audit_list
                            subproc_list = proc_list[start_idx : end_idx + 1]
                            # get basename of uploaded file
                            basename = uploaded_file.name.split(".")[0]
                            # save subproc_list to file using upload
                            savedf(subproc_list, basename)

                    else:
                        st.error("不支持文件类型")

        # display all policy
        st.write("已编码的文件：")
        uploadfilels = get_uploadfiles()
        # st.write(uploadfilels)
        # display all upload files
        for uploadfile in uploadfilels:
            st.markdown(f"- {uploadfile}")
        remove = st.button("删除已上传文件")
        if remove:
            remove_uploadfiles()
            st.success("删除成功")

    elif choice == "文件选择":
        # choose radio for file1 or file2
        file_choice = st.sidebar.radio("选择文件", ["选择文件1", "选择文件2"])

        if file_choice == "选择文件1":
            # get current file1 value from session
            industry_choice = st.session_state["file1_industry"]
            rule_choice = st.session_state["file1_rulechoice"]
            filetype_choice = st.session_state["file1_filetype"]
            section_choice = st.session_state["file1_section_list"]
        elif file_choice == "选择文件2":
            # get current file2 value from session
            industry_choice = st.session_state["file2_industry"]
            rule_choice = st.session_state["file2_rulechoice"]
            filetype_choice = st.session_state["file2_filetype"]
            section_choice = st.session_state["file2_section_list"]
        # get preselected filetype index
        file_typels = ["审计程序", "已上传文件"]
        if filetype_choice == "":
            filetype_index = 0
        else:
            filetype_index = file_typels.index(filetype_choice)
        # choose file type
        file_type = st.sidebar.selectbox(
            "选择文件类型",
            file_typels,
            index=filetype_index,
        )
        
        if file_type == "审计程序":

            industry_list = get_folder_list(auditfolder)

            # get preselected industry index
            # if industry_choice in industry_list:
            #     industry_index = industry_list.index(industry_choice)
            # else:
            industry_index = 0
            rule_choice = []
            section_choice = []

            industry_choice = st.sidebar.selectbox(
                "选择行业:", industry_list, index=industry_index
            )

            

        elif file_type == "已上传文件":
            if industry_choice != "":
                rule_choice = []
            upload_list = get_uploadfiles()
            upload_choice = st.sidebar.multiselect("选择已上传文件:", upload_list, rule_choice)
            if upload_choice != []:
                choosedf, choose_embeddings = get_upload_data(upload_choice)
            else:
                choosedf, choose_embeddings = None, None
            industry_choice = ""
            rule_choice = upload_choice
            rule_column_ls = []

        # file choose button
        file_button = st.sidebar.button("选择文件")
        if file_button:
            if file_choice == "选择文件1":
                file1df, file1_embeddings = choosedf, choose_embeddings
                file1_industry = industry_choice
                file1_rulechoice = rule_choice
                file1_filetype = file_type
                file1_section_list = rule_column_ls
                st.session_state["file1df"] = file1df
                st.session_state["file1_embeddings"] = file1_embeddings
                st.session_state["file1_industry"] = file1_industry
                st.session_state["file1_rulechoice"] = file1_rulechoice
                st.session_state["file1_filetype"] = file1_filetype
                st.session_state["file1_section_list"] = file1_section_list
                file2df = st.session_state["file2df"]
                file2_embeddings = st.session_state["file2_embeddings"]
                file2_industry = st.session_state["file2_industry"]
                file2_rulechoice = st.session_state["file2_rulechoice"]
                file2_filetype = st.session_state["file2_filetype"]
                file2_section_list = st.session_state["file2_section_list"]
            elif file_choice == "选择文件2":
                file2df, file2_embeddings = choosedf, choose_embeddings
                file2_industry = industry_choice
                file2_rulechoice = rule_choice
                file2_filetype = file_type
                file2_section_list = rule_column_ls
                st.session_state["file2df"] = file2df
                st.session_state["file2_embeddings"] = file2_embeddings
                st.session_state["file2_industry"] = file2_industry
                st.session_state["file2_rulechoice"] = file2_rulechoice
                st.session_state["file2_filetype"] = file2_filetype
                st.session_state["file2_section_list"] = file2_section_list
                file1df = st.session_state["file1df"]
                file1_embeddings = st.session_state["file1_embeddings"]
                file1_industry = st.session_state["file1_industry"]
                file1_rulechoice = st.session_state["file1_rulechoice"]
                file1_filetype = st.session_state["file1_filetype"]
                file1_section_list = st.session_state["file1_section_list"]
        else:
            file1df = st.session_state["file1df"]
            file2df = st.session_state["file2df"]
            file1_embeddings = st.session_state["file1_embeddings"]
            file2_embeddings = st.session_state["file2_embeddings"]
            file1_industry = st.session_state["file1_industry"]
            file2_industry = st.session_state["file2_industry"]
            file1_rulechoice = st.session_state["file1_rulechoice"]
            file2_rulechoice = st.session_state["file2_rulechoice"]
            file1_filetype = st.session_state["file1_filetype"]
            file2_filetype = st.session_state["file2_filetype"]
            file1_section_list = st.session_state["file1_section_list"]
            file2_section_list = st.session_state["file2_section_list"]

        # file choose reset
        file_reset = st.sidebar.button("重置文件")
        if file_reset:
            file1df, file2df = None, None
            file1_embeddings, file2_embeddings = None, None
            file1_industry, file2_industry = "", ""
            file1_rulechoice, file2_rulechoice = [], []
            file1_filetype, file2_filetype = "", ""
            file1_section_list, file2_section_list = [], []
            st.session_state["file1df"] = file1df
            st.session_state["file2df"] = file2df
            st.session_state["file1_embeddings"] = file1_embeddings
            st.session_state["file2_embeddings"] = file2_embeddings
            st.session_state["file1_industry"] = file1_industry
            st.session_state["file2_industry"] = file2_industry
            st.session_state["file1_rulechoice"] = file1_rulechoice
            st.session_state["file2_rulechoice"] = file2_rulechoice
            st.session_state["file1_filetype"] = file1_filetype
            st.session_state["file2_filetype"] = file2_filetype
            st.session_state["file1_section_list"] = file1_section_list
            st.session_state["file2_section_list"] = file2_section_list

        st.subheader("已选择的文件1：")
        # display file1 rulechoice
        if file1_rulechoice != []:
            # convert to string
            file1_rulechoice_str = "| ".join(file1_rulechoice)
            # display string
            st.warning("文件1：" + file1_rulechoice_str)
        else:
            st.error("文件1：无")
        # display file1 section
        if file1_section_list != []:
            # convert to string
            file1_section_str = "| ".join(file1_section_list)
            # display string
            st.info("章节1：" + file1_section_str)
        else:
            st.info("章节1：全部")

        st.subheader("已选择的文件2：")
        # display file2 rulechoice
        if file2_rulechoice != []:
            # convert to string
            file2_rulechoice_str = "| ".join(file2_rulechoice)
            # display string
            st.warning("文件2：" + file2_rulechoice_str)
        else:
            st.error("文件2：无")

        # display file2 section
        if file2_section_list != []:
            # convert to string
            file2_section_str = "| ".join(file2_section_list)
            # display string
            st.info("章节2：" + file2_section_str)
        else:
            st.info("章节2：全部")


    elif choice == "文件问答":
        st.subheader("文件问答")
        # enbedding button
        embedding = st.sidebar.button("生成问答模型")
        if embedding:
            with st.spinner("正在生成问答模型..."):
                # generate embeddings
                build_index()
                st.success("问答模型生成完成")

        # question input
        question = st.text_input("输入问题")
        if question != "":
            # answer button
            answer_btn = st.button("获取答案")
            if answer_btn:
                with st.spinner("正在获取答案..."):
                    # get answer
                    answer = gpt_answer(question)
                    st.write(answer)


if __name__ == "__main__":
    main()
