import pandas as pd
import streamlit as st

from gptfuc import build_index, gpt_answer
from upload import (
    copy_files,
    get_uploadfiles,
    remove_uploadfiles,
    save_uploadedfile,
    savedf,
)
from utils import get_wpdf  # df2aggrid,

uploadfolder = "uploads"
filerawfolder = "fileraw"


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
                            # wp save button
                            save_btn = st.sidebar.button("保存底稿")
                            if save_btn:
                                # get basename of uploaded file
                                basename = uploaded_file.name.split(".")[0]
                                # save subproc_list to file using upload
                                savedf(subproc_list, basename)
                                st.sidebar.success("保存成功")

                    else:
                        st.error("不支持文件类型")

        # display all upload files
        st.write("已上传的文件：")
        uploadfilels = get_uploadfiles(uploadfolder)
        # st.write(uploadfilels)
        # display all upload files
        for uploadfile in uploadfilels:
            st.markdown(f"- {uploadfile}")
        remove = st.button("删除已上传文件")
        if remove:
            remove_uploadfiles(uploadfolder)
            st.success("删除成功")

    elif choice == "文件选择":
        # intialize session state
        if "wp_choice" not in st.session_state:
            st.session_state["wp_choice"] = ""
        if "file_list" not in st.session_state:
            st.session_state["file_list"] = []
        # choose radio for file1 or file2
        file_choice = st.sidebar.radio("选择文件", ["选择文件", "选择底稿"])

        if file_choice == "选择底稿":
            # get current wp_choice value from session
            wp_choice = st.session_state["wp_choice"]
            upload_list = get_uploadfiles(uploadfolder)
            upload_choice = st.sidebar.selectbox("选择已上传文件:", upload_list)

            if upload_choice == []:
                st.error("请选择文件")
                return
            wp_choice = upload_choice

        elif file_choice == "选择文件":
            # get current file2 value from session
            file_list = st.session_state["file_list"]

            upload_list = get_uploadfiles(uploadfolder)
            upload_choice = st.sidebar.multiselect("选择已上传文件:", upload_list, file_list)

            if upload_choice == []:
                st.error("请选择文件")
                return

            file_list = upload_choice

        # file choose button
        file_button = st.sidebar.button("选择文件")
        if file_button:
            if file_choice == "选择文件":
                st.session_state["file_list"] = file_list
                wp_choice = st.session_state["wp_choice"]
                # copy file_list to filerawfolder
                copy_files(file_list, uploadfolder, filerawfolder)
            elif file_choice == "选择底稿":
                st.session_state["wp_choice"] = wp_choice
                file_list = st.session_state["file_list"]
        else:
            file_list = st.session_state["file_list"]
            wp_choice = st.session_state["wp_choice"]

        # file choose reset
        file_reset = st.sidebar.button("重置文件")
        if file_reset:
            file_list = []
            wp_choice = ""
            st.session_state["file_list"] = file_list
            st.session_state["wp_choice"] = wp_choice

        # enbedding button
        embedding = st.sidebar.button("生成问答模型")
        if embedding:
            # with st.spinner("正在生成问答模型..."):
            # generate embeddings
            try:
                build_index()
                st.success("问答模型生成完成")
            except Exception as e:
                st.error(e)
                st.error("问答模型生成失败，请检查文件格式")

        st.subheader("已选择的文件：")
        # display file1 rulechoice
        if file_list != []:
            # convert to string
            file_choice_str = "| ".join(file_list)
            # display string
            st.warning("文件：" + file_choice_str)
        else:
            st.error("文件：无")

        st.subheader("已选择的底稿：")
        # display file2 rulechoice
        if wp_choice != "":
            # display string
            st.warning("底稿" + wp_choice)
        else:
            st.error("底稿：无")

        # display all upload files
        st.write("待编码的文件：")
        uploadfilels = get_uploadfiles(filerawfolder)
        # display all upload files
        for uploadfile in uploadfilels:
            st.markdown(f"- {uploadfile}")
        remove = st.button("删除待编码的文件")
        if remove:
            remove_uploadfiles(filerawfolder)
            st.success("删除成功")
    elif choice == "文件问答":
        st.subheader("文件问答")

        mode = st.sidebar.radio("选择模式", ["单条", "批量"])

        if mode == "单条":
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
        elif mode == "批量":
            wp_choice = st.session_state["wp_choice"]

            # input prompt text
            prompt_text = st.sidebar.text_area("输入提示文本")
            # st.subheader("已选择的底稿：")
            # display file2 rulechoice
            if wp_choice != "":
                # display string
                st.sidebar.warning("底稿" + wp_choice)
                wpname = wp_choice.split(".")[0]
                wpdf = get_wpdf(wpname, uploadfolder)
                wplen = len(wpdf)
                # choose page start number and end number
                start_num = st.sidebar.number_input("起始页", value=0, min_value=0)
                # convert to int
                start_num = int(start_num)
                end_num = st.sidebar.number_input("结束页", value=wplen - 1)
                # convert to int
                end_num = int(end_num)
                subwpdf = wpdf[start_num : end_num + 1]
                questionls = subwpdf["条款"].tolist()
                st.write(questionls)
                # answer button
                answer_btn = st.button("获取答案")
                if answer_btn:
                    with st.spinner("正在获取答案..."):
                        for idx, question in enumerate(questionls):
                            full_question = prompt_text + question
                            st.write("问题" + str(idx) + "： " + full_question)
                            # get answer
                            answer = gpt_answer(full_question)
                            st.write(answer)
            else:
                st.sidebar.error("请选择底稿")


if __name__ == "__main__":
    main()
