import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re



url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQdYosYor7k_lxER-u8UUyIYwUbevb8Ygr-xO7fT-coTtkXt8Owh-BL_PuJ3jnkWWOyY6wXbt1z1GGx/pub?gid=1365251745&single=true&output=csv'

dat = pd.read_csv(url)

dat['catprof'] = dat.iloc[:,4].fillna('')+dat.iloc[:,7].fillna('')
profile = dat['catprof'].values
vectorizer = CountVectorizer(ngram_range=(2,5), analyzer='char')
X = vectorizer.fit_transform(profile)
# X = X>0



def getrank(query):
  res = vectorizer.transform([query])
  # res = res>0
  matchscore = X.dot(res.T).toarray()[:,0]/(np.diff(X.indptr)+0.001)
  # sortind = np.argsort(sortscore)[::-1]
  return matchscore

def getmatch(query,mail):
    res = getrank(query)
    rdat = dat.copy()

    rdat['score'] = res
    rdat = rdat[rdat['Email address']!=mail]
    rdat = rdat.sort_values(by='score',ascending=False)
    return rdat.iloc[0:3,[4,7,1,-1]]

# Define function to display a pandas DataFrame or Series with wrapped text
def display_data(df):
    st.write(
        """
    <style>
    .dataframe tbody tr td {
        white-space: pre-wrap;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Display DataFrame
    st.dataframe(df)

def display_html(resList,cate,memd):

    st.markdown(
        """
    <style>
    table {
      font-family: sans-serif;
      border-collapse: collapse;
      width: 100%;
      background-color: #000000;
    }

    td, th {
      border: 1px solid #ffffff;
      text-align: left;
      padding: 8px;
      color: #ffffff;
      font-weight: normal;
    }

    tr:nth-child(even) {
      background-color: #222222;
    }

    tr:nth-child(odd) {
      background-color: #000000;
    }

    tr:hover {
      background-color: #444444;
    }

    .stTextInput input {
      border: 2px solid #2e3d49;
      border-radius: 5px;
      padding: 8px 12px;
      font-size: 14px;
    }

    .stTextInput input:focus {
      outline: none;
      border-color: #4d7bb7;
      box-shadow: 0 0 5px #4d7bb7;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    res = ''
    for r in resList:
        res = res + '<tr><th>%s</th><th>%s</th><th>%s</th><th>%.1f</th></tr>'%(r[0],r[1],r[2],r[3]*100)

    html = '''

        <tr>
            <th>สนใจแข่งหมวด</th>
            <th>โปรไฟล์</th>
            <th>E-mail</th>
            <th>คะแนนตรงตามต้องการ</th>
        </tr>
    '''
    header = '''<p>คุณสนใจลงแข่งหมวด:<br> %s</p><p>สมาชิกที่ต้องการ:<br> %s</p>'''%(cate,memd)
    st.markdown(header, unsafe_allow_html=True)

    html_content = html + res
    st.markdown(html_content, unsafe_allow_html=True)


def app():
    mail = st.text_input('กรุณาใส่ E-mail ของคุณที่ลงทะเบียนหาทีม', 'ใส่ E-mail ...')
    mail = mail.strip()
    query = dat[dat['Email address']==mail]
    
    if len(query)>0:
        cate = query['สนใจแข่งหมวดไหน'].values[0]
        memd = query['กรุณาเขียนเล่าถึงทีมงานที่อยากได้ (เพื่อการ matching ที่ตรงกับความต้องการ กรุณากรอกข้อมูลในส่วนนี้ให้มากที่สุด)'].values[0]
        query = cate+memd

        # st.text(query)


        df = getmatch(query,mail)
        # return df

        # display_data(df)
        # resList = ['hello','world']
        display_html(df.values,cate,memd)


# call the Streamlit app
if __name__ == "__main__":
    app()
