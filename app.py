import pandas as pd
import streamlit as st

from gptfuc import (  # gpt_vectoranswer,; local_search,
    add_to_index,
    build_index,
    gpt_auditanswer,
)
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
    menu = ["底稿上传", "文件编码", "文件选择", "智能匹配"]
    choice = st.sidebar.selectbox("选择", menu)

    if choice == "底稿上传":
        st.subheader("底稿上传")
        # choose input method of manual or upload file
        # input_method = st.sidebar.radio("文件上传方式", ["上传底稿"])

        # if input_method == "项目文件":
        #     uploaded_file_ls = st.file_uploader(
        #         "选择新文件上传",
        #         type=["docx", "pdf", "txt"],
        #         accept_multiple_files=True,
        #         help="选择文件上传",
        #     )
        #     for uploaded_file in uploaded_file_ls:
        #         if uploaded_file is not None:
        #             # Check File Type
        #             if (
        #                 (
        #                     uploaded_file.type
        #                     == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        #                 )
        #                 | (uploaded_file.type == "application/pdf")
        #                 | (uploaded_file.type == "text/plain")
        #             ):
        #                 save_uploadedfile(uploaded_file)

        # if input_method == "上传底稿":
        uploaded_file_ls = st.file_uploader(
            "选择新文件上传",
            type=["xlsx"],
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
        # st.write("已上传的文件：")
        # uploadfilels = get_uploadfiles(uploadfolder)
        # # st.write(uploadfilels)
        # # display all upload files
        # for uploadfile in uploadfilels:
        #     st.markdown(f"- {uploadfile}")
        # remove = st.button("删除已上传文件")
        # if remove:
        #     remove_uploadfiles(uploadfolder)
        #     st.success("删除成功")

    elif choice == "文件选择":
        # intialize session state
        if "wp_choice" not in st.session_state:
            st.session_state["wp_choice"] = ""
        if "file_list" not in st.session_state:
            st.session_state["file_list"] = []

        # if "docsearch" not in st.session_state:
        #     st.session_state["docsearch"] = None
        # choose radio for file1 or file2
        file_choice = st.sidebar.radio("选择文件", ["选择文件", "选择底稿"])

        if file_choice == "选择底稿":
            # get current wp_choice value from session
            # wp_choice = st.session_state["wp_choice"]
            upload_list = get_uploadfiles(uploadfolder)
            upload_choice = st.sidebar.selectbox("选择已上传文件:", upload_list)

            if upload_choice == []:
                st.error("请选择文件")
                # return
            wp_choice = upload_choice

        elif file_choice == "选择文件":
            # get current file2 value from session
            # file_list = st.session_state["file_list"]

            upload_list = get_uploadfiles(filerawfolder)
            upload_choice = st.sidebar.multiselect("选择已上传文件:", upload_list)

            if upload_choice == []:
                st.error("请选择文件")
                # return

            file_list = upload_choice

        # file choose button
        file_button = st.sidebar.button("选择文件")
        if file_button:
            if file_choice == "选择文件":
                st.session_state["file_list"] = file_list
                wp_choice = st.session_state["wp_choice"]
                # copy file_list to filerawfolder
                # copy_files(file_list, uploadfolder, filerawfolder)
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

    elif choice == "文件编码":
        upload_list = get_uploadfiles(uploadfolder)
        upload_choice = st.sidebar.multiselect("选择已上传文件:", upload_list)

        if upload_choice == []:
            st.error("请选择文件")
            # return

        file_list = upload_choice

        # file choose button
        file_button = st.sidebar.button("选择文件")
        if file_button:
            # copy file_list to filerawfolder
            copy_files(file_list, uploadfolder, filerawfolder)

        # enbedding button
        embedding = st.sidebar.button("重新生成模型")
        if embedding:
            # with st.spinner("正在生成问答模型..."):
            # generate embeddings
            try:
                build_index()
                st.success("问答模型生成完成")
            except Exception as e:
                st.error(e)
                st.error("问答模型生成失败，请检查文件格式")

        # add documnet button
        add_doc = st.sidebar.button("模型添加文档")
        if add_doc:
            # with st.spinner("正在添加文档..."):
            # generate embeddings
            try:
                add_to_index()
                st.success("文档添加完成")
            except Exception as e:
                st.error(e)
                st.error("文档添加失败，请检查文件格式")
        remove = st.button("删除待编码的文件")
        if remove:
            remove_uploadfiles(filerawfolder)
            st.success("删除成功")
        # display all upload files
        st.write("待编码的文件：")
        uploadfilels = get_uploadfiles(filerawfolder)
        # display all upload files
        for uploadfile in uploadfilels:
            st.markdown(f"- {uploadfile}")

        # st.sidebar.subheader("删除所有文件")
        remove = st.sidebar.button("删除已上传文件")
        if remove:
            remove_uploadfiles(uploadfolder)
            st.success("删除成功")

    elif choice == "智能匹配":
        mode = st.sidebar.selectbox("选择模式", ["单一", "批量"])

        file_list = st.session_state["file_list"]
        # choose chain type
        # chain_type = st.sidebar.selectbox(
        #     "选择链条类型", ["stuff", "map_reduce", "refine", "map_rerank"]
        # )
        chain_type = "stuff"
        # choose model
        model_name = st.sidebar.selectbox(
            "选择模型",
            [
                "gpt-35-turbo",
                "gpt-35-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
                "gpt-4-turbo",
                "tongyi",
                "ERNIE-Bot-4",
                "ERNIE-Bot-turbo",
                "ChatGLM2-6B-32K",
                "Yi-34B-Chat",
                "gemini-pro",
            ],
        )
        # choose top_k
        top_k = st.sidebar.slider("选择top_k", 1, 10, 3)
        if mode == "单一":
            st.subheader("单一匹配")
            # choose the task option
            # task = st.sidebar.radio("选择任务", ["问答", "审计"])
            # if task == "问答":
            #     # question input
            #     question = st.text_area("输入问题")

            #     # answer button
            #     answer_btn = st.button("获取答案")
            #     if answer_btn:
            #         if question != "" and chain_type != "":
            #             with st.spinner("正在获取答案..."):
            #                 # get answer
            #                 # answer = gpt_answer(question,chain_type)
            #                 # docsearch = st.session_state["docsearch"]
            #                 answer, sourcedb = gpt_vectoranswer(
            #                     question, chain_type, top_k=top_k, model_name=model_name
            #                 )
            #                 st.markdown("#### 答案")
            #                 st.write(answer)
            #                 with st.expander("查看来源"):
            #                     st.markdown("#### 来源")
            #                     st.table(sourcedb)
            #         else:
            #             st.error("问题或链条类型不能为空")
            # elif task == "审计":
            # question input
            question = st.text_area("审计要求")

            # answer button
            answer_btn = st.button("获取答案")
            if answer_btn:
                if question != "" and chain_type != "":
                    with st.spinner("正在获取答案..."):
                        # get answer
                        # answer = gpt_answer(question,chain_type)
                        # docsearch = st.session_state["docsearch"]
                        answer, sourcedb = gpt_auditanswer(
                            question,
                            file_list,
                            chain_type,
                            top_k=top_k,
                            model_name=model_name,
                        )
                        st.markdown("#### 结果")
                        st.write(answer)
                        with st.expander("查看来源"):
                            st.markdown("#### 来源")
                            st.table(sourcedb)
                else:
                    st.error("问题或链条类型不能为空")

        elif mode == "批量":
            st.subheader("批量匹配")
            wp_choice = st.session_state["wp_choice"]

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
                # st.write(questionls)
                # display question
                st.write("问题列表：")
                for idx, question in enumerate(questionls):
                    st.write(str(idx) + "： " + question)

                # resultls to save the result
                resultls = []
                # answer button
                answer_btn = st.button("获取答案")
                if answer_btn:
                    with st.spinner("正在获取答案..."):
                        for idx, question in enumerate(questionls):
                            st.markdown("#### 问题" + str(idx) + "： " + question)
                            answer, sourcedb = gpt_auditanswer(
                                question,
                                file_list,
                                chain_type,
                                top_k=top_k,
                                model_name=model_name,
                            )
                            st.markdown("#### 答案" + str(idx) + "：")
                            st.write(answer)
                            with st.expander("查看来源"):
                                st.markdown("#### 来源")
                                st.table(sourcedb)
                            resultls.append(answer)

                        # save resultls to csv
                        resultdf = pd.DataFrame({"问题": questionls, "答案": resultls})
                        st.sidebar.download_button(
                            label="下载结果",
                            data=resultdf.to_csv(index=False),
                            file_name="result.csv",
                            mime="text/csv",
                        )

            else:
                st.sidebar.error("请选择底稿")


if __name__ == "__main__":
    main()
