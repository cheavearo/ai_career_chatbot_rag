import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import streamlit as st

class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message

        if usecase == "rag_chatbot":
            # 1️⃣ Prepare initial state
            initial_state = {"question": user_message}

            # 2️⃣ Invoke LangGraph
            res = graph.invoke(initial_state)
            print("GRAPH RESULT:", res)  # Debugging print

            # 3️⃣ Display user input
            with st.chat_message("user"):
                st.write(user_message)

            # 4️⃣ Display retrieved documents (if any)
            if "documents" in res and res["documents"]:
                with st.chat_message("ai"):
                    st.write("📄 Retrieved Context:")
                    for i, doc in enumerate(res["documents"], start=1):
                        st.markdown(f"**Doc {i}:** {doc.page_content[:300]}...")

            # 5️⃣ Display final answer
            answer = res.get("final_answer") or res.get("generation") or "No answer generated."
            with st.chat_message("assistant"):
                st.write(answer)






