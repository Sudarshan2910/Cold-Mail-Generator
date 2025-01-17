from chains import Chain
from portfolio import Portfolio
from utils import clean_text
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader


def create_streamlit(llm,portfolio,clean_text):

    st.title("📧 Cold Mail Generator")
    url_input=st.text_input("Enter a URL: ",value="https/microsoft.com")
    submit_button=st.button("Submit")

    if submit_button:
        try:
            loader=WebBaseLoader([url_input])
            data=clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs=llm.extract_jobs(data)
            for job in jobs:
                skills=job.get('skills',[])
                link=portfolio.query_links(skills)
                email=llm.extract_mail(job,link)
                st.code(email,language='markdown')

        except Exception as e:
            st.error(f"An error Occurred: {e}")
            


if __name__=="__main__":
    chain=Chain()
    portfolio=Portfolio()
    st.set_page_config(layout="wide",page_title="COLD MAIL GENERATOR", page_icon="📧")
    create_streamlit(chain,portfolio,clean_text)