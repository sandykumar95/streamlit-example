import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import math

#@st.cache(suppress_st_warning=True)
def pretoss(t1,t2):
        old_matches = pd.read_csv('matches.csv')
        #old_matches
        #print("Data Frame read")
        
        sample=old_matches.drop(['id','season','city','date','result','dl_applied','win_by_runs','win_by_wickets','player_of_match','venue','umpire1','umpire2','umpire3'],axis=1)
        #sample
        #print("dropping other rows")
        
        a = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
        'Rising Pune Supergiant', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
        'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
        'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants', 'Delhi Capitals']
        b = ['SRH','MI','GL','RPS','RCB','KKR','DC','KXIP','CSK','RR','SRH','KTK','PW','RPS','DC']
        
        sample.replace(a,b,inplace = True)
        #sample
        #print("Renamed Teams")
        sample=sample.dropna()
        #sample
        
        sample = sample[sample.team1 != 'KTK']
        sample = sample[sample.team1 != 'RPS']
        sample = sample[sample.team1 != 'PW']
        sample = sample[sample.team1 != 'GL']
        #sample = sample[sample.team1 != 'SRH1']#KTK RPS PW GL
        #sample
        
        sample = sample[sample.team2 != 'KTK']
        sample = sample[sample.team2 != 'RPS']
        sample = sample[sample.team2 != 'PW']
        sample = sample[sample.team2 != 'GL']#KTK RPS PW GL
        #sample = sample[sample.team2 != 'SRH1']
        #sample
        #print("Removed non existing teams")
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        
        sample = pd.get_dummies(sample, prefix=['Team_1', 'Team_2'], columns=['team1', 'team2'])
        #sample
        #print("Created Dummies")
        
        X = sample.drop(['winner','toss_decision','toss_winner'], axis=1)
        y = sample["winner"]
        
        # Separate train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=random.uniform(0.4, 0.35), random_state=42)
        #X_train
        #print("Separate train and test sets")
        
        rf = RandomForestClassifier(n_estimators=100,oob_score=True ,class_weight='balanced',verbose=2,n_jobs=-1, max_depth=40,random_state=42)
        rf.fit(X_train, y_train)
        #print("Applied RandomForestClassifier")
        
        score = rf.score(X_train, y_train)
        score2 = rf.score(X_test, y_test)
        print(score)
        print(score2)
        print(rf.oob_score_)
        
        copy=pd.read_csv('2020 Copy.csv')
        #copy
        
        c=['Sunrisers hyderabad', 'Mumbai indians', 'Gujarat Lions',
        'Rising Pune Supergiant', 'Royal challengers bangalore',
        'Kolkata knight riders', 'Delhi Daredevils', 'Kings xi punjab',
        'Chennai super kings', 'Rajasthan royals', 'Deccan Chargers',
        'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants', 'Delhi capitals','Royal challengers']
        d= ['SRH','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','SRH','KTK','PW','RPS','DC','RCB']
        
        copy.replace(c,d,inplace = True)
        #copy
        
        et1=list(copy['Team'])
        et2=list(copy['Team2'])
        
        copy = pd.get_dummies(copy, prefix=['Team_1', 'Team_2'], columns=['Team', 'Team2'])
        #copy
        
        predictions=rf.predict(copy)
        #print(predictions)
        print(len(predictions))
        
        print(t1,t2)
        #print(et1,et2)
        #print(len(et1),len(et2))
        
        for i in range(56):
            if(et1[i]==t1 and et2[i]==t2):
                print(predictions[i])
                winner_is=predictions[i]
                if(t1!=winner_is):
                    looser_is=t1
                else:
                    looser_is=t2
                st.success(predictions[i]+" will win")
        
        score2=score2*100
        k=math.ceil(score2)
        fig = go.Figure(data=[go.Pie(labels=[winner_is,looser_is], textinfo='label+percent',values=[k,100-k], hole=.2)])
        fig.update_layout(height=600, title_text="Pre-Toss Sims", font=dict(family='Courier New, monospace', size=18, color='#000000'))
        st.plotly_chart(fig,use_container_width=True)
        st.write("### The Pre-Toss ML model predicted that there is "+str(k)+"% chance for "+winner_is+" will win the game")
        
        return score2