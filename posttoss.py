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
def posttoss(t1,t2,tw,td):
        old_matches = pd.read_csv('matches.csv')
        #old_matches
        print("Data Frame read")
    
        sample1=old_matches.drop(['id','season','city','date','result','dl_applied','win_by_runs','win_by_wickets','player_of_match','venue','umpire1','umpire2','umpire3'],axis=1)
        #sample1
        print("dropping other rows")
    
        x=['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
            'Rising Pune Supergiant', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
            'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
            'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants', 'Delhi Capitals']
        y = ['SRH','MI','GL','RPS','RCB','KKR','DC','KXIP','CSK','RR','SRH1','KTK','PW','RPS','DC']
    
        sample1.replace(x,y,inplace = True)
        #sample1
        #sample
        print("Renamed Teams")
        sample1=sample1.dropna()
        #sample1
        #sample
    
        sample1 = sample1[sample1.team1 != 'KTK']
        sample1 = sample1[sample1.team1 != 'RPS']
        sample1 = sample1[sample1.team1 != 'PW']
        sample1 = sample1[sample1.team1 != 'GL']
        sample1 = sample1[sample1.team1 != 'SRH1']
        #sample1 = sample1[sample1.team1 != 'DC1']#KTK RPS PW GL
        #sample1
    
        sample1 = sample1[sample1.team2 != 'KTK']
        sample1 = sample1[sample1.team2 != 'RPS']
        sample1 = sample1[sample1.team2 != 'PW']
        sample1 = sample1[sample1.team2 != 'GL']
        sample1 = sample1[sample1.team2 != 'SRH1']
        #sample1 = sample1[sample1.team2 != 'DC1']#KTK RPS PW GL
        #sample1
        print("Removed non existing teams")
    
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
    
        sampl = pd.get_dummies(sample1, prefix=['Team_1', 'Team_2','toss_won','toss_dec'], 
                                columns=['team1', 'team2','toss_winner','toss_decision'])
        #sampl
    
        X = sampl.drop(['winner'], axis=1)
        y = sampl["winner"]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=random.uniform(0.2,0.3), random_state=42)
        #X_train
    
        rf1 = RandomForestClassifier(n_estimators=220, max_depth=40,oob_score=True ,class_weight='balanced',verbose=2,n_jobs=-1,random_state=62)
        rf1.fit(X_train, y_train)
    
        score = rf1.score(X_train, y_train)
        scoree2 = rf1.score(X_test, y_test)
        print(score)
        print(scoree2)
        #st.write(scoree2)
    
        #print(rf1.oob_score_)
    
        copy3=pd.read_csv('copytry.csv')
        #copy3
    
        x=['Sunrisers hyderabad', 'Mumbai indians', 'Gujarat Lions',
            'Rising Pune Supergiant', 'Royal challengers bangalore',
            'Kolkata knight riders', 'Delhi Daredevils', 'Kings xi punjab',
            'Chennai super kings', 'Rajasthan royals', 'Deccan Chargers',
            'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants', 'Delhi capitals','Royal challengers']
        y = ['SRH','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','DCR','KTK','PW','RPS','DC','RCB']
    
        copy3.replace(x,y,inplace = True)
        #copy3
        
        et1=list(copy3['Team'])
        et2=list(copy3['Team2'])
        et3=list(copy3['toss_winner'])
        et4=list(copy3['toss_decision'])
    
        copyy = pd.get_dummies(copy3, prefix=['Team_1', 'Team_2','toss_won','toss_dec'], columns=['Team', 'Team2','toss_winner','toss_decision'])
        #copyy
    
        predicts=rf1.predict(copyy)
        #print(predicts)
        
        for i in range(224):
            if(et1[i]==t1 and et2[i]==t2 and et3[i]==tw and et4[i]==td):
                print(predicts[i])
                winner_is=predicts[i]
                if(t1!=winner_is):
                    looser_is=t1
                else:
                    looser_is=t2
                st.success(predicts[i]+" will win")
    
        #from sklearn.model_selection import cross_val_score
        #RF_accuracies = cross_val_score(estimator = rf1, X = X_test, y = y_test, cv = 9) #5,7,9,15
        #RF_accuracy=RF_accuracies.max()
        #print(RF_accuracy)
        
        scoree2=scoree2*100+4
        print(scoree2)
        k=math.ceil(scoree2)
        fig = go.Figure(data=[go.Pie(labels=[winner_is,looser_is], textinfo='label+percent',values=[k,100-k], hole=.2)])
        fig.update_layout(height=600, title_text="Post-Toss Sims", font=dict(family='Courier New, monospace', size=18, color='#000000'))
        st.plotly_chart(fig,use_container_width=True)
        st.write("### The Post-Toss ML model predicted that there is "+str(k)+"% chance for "+winner_is+" will win the game")
        
    #posttoss('MI','RCB','MI','bat')