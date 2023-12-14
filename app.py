import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Response from LLAma 2 model
def getResponseLLAma(input_text,no_words,content_user):
    llm=CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})

    template="""
            Write a content for {content_user} for topic {input_text} within {no_words} words.
            """

    prompt=PromptTemplate(input_variables=["content_user","input_text","no_words"],
                        template=template)

    response=llm(prompt.format(content_user=content_user,input_text=input_text,no_words=no_words))
    print(response)
    return response

st.set_page_config(page_title='Content Creation',
                   page_icon='\N{grinning face}',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header('Generate Content')
input_text=st.text_input('Enter the Content Topic')

col1,col2= st.columns([5,5])
with col1:
    no_words=st.text_input('No. of words')
with col2:
    content_user=st.selectbox('Writing the content for', ('Student','Researchers','Scientists', 'General people'),index=0)

submit=st.button('Generate the content')

if submit:
    st.write(getResponseLLAma(input_text,no_words,content_user))
