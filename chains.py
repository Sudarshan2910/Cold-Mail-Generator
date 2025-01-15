import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()


class Chain:
    def __init__(self):
        self.llm=ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-70b-versatile"
        )
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ###Scraping text from website:
            {page_data}
            ###Instruction:
            The scraped data is from the career's page of website.
            Your job is to extract the job postings and return them in json format containing the
            following keys: 'role', 'experience', 'skills' and 'description'.
            Only return the valid JSON
            ###Valid JSON (No Preamble)
            """
        )
        chain_extract= prompt_extract | self.llm  
        res=chain_extract.invoke(input={'page_data':cleaned_text})
        try:
            json_parser= JsonOutputParser()
            res=json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res,list) else [res]
    
    def extract_mail(self,job,links):
        prompt_email=PromptTemplate.from_template(
            """
            ###Job Description:
            {job_description}
            ###INSTRUCTION:
            You are Sudarshan Goyal, a final year btech student  specializing in Computer Science and Engineering with a focus on AI and Analytics.
            Write a professional and concise cold email introducing myself to the client regarding the job mentioned above in fulfilling there needs.
            Also add the most relevant project links to showcase my potential to them:{link_list}
            Do not provide a preamble
            ###Email (NO PREAMBLE)
            """
        )
        chain_email=prompt_email | self.llm
        res=chain_email.invoke({"job_description":str(job),"link_list":links})
        return res.content

if __name__=="__main__":
    print(os.getenv("GROQ_API_KEY"))
