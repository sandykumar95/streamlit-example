import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import math

from pretoss import*
from posttoss import*


st.title('Welcome to IPL Predictions & Analysis APP')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            footer:before {
        	content:'⭐️ Thankyou for visiting this site ❤️'; 
        	visibility: visible;
        	display: block;
        	position: relative;
        	#background-color: red;
        	padding: 5px;
            color: black;
        	top: 2px;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


#page_bg_img = '''<style>body {background-image: url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e");background-size: cover;}</style>'''
#st.markdown(page_bg_img, unsafe_allow_html=True)
#st.image('virat-1.jpg', output_format='PNG',use_column_width=True)
#st.audio('bahubali_bgm.mp3')
#https://unsplash.com/photos/bY4cqxp7vos

q=st.selectbox("Select Model",['Select','IPL Data Analytics 📈','Pre-Toss Prediction 🏏', 'Post-Toss Prediction 🏏'])
#st.warning('Please input a name.')


if(q=='Pre-Toss Prediction 🏏'):
    st.title('Pre-Toss Prediction')
    k=st.selectbox("Team 1",['RCB', 'MI', 'SRH', 'CSK', 'KKR', 'DC', 'RR', 'KXIP'])
    j=st.selectbox("Team 2",['MI', 'SRH', 'CSK', 'KKR', 'DC', 'RR', 'KXIP', 'RCB'])
    if(k==j):
        st.error("Team Names should be different")
    st.write("**Note**:  "+k+" vs "+j+" is not same as "+j+" vs "+k)
    st.subheader(k+" vs "+j)
    
    #me=st.button("Predict")
    me, col2 = st.beta_columns([1,0.225])
    
    with me:
        me=st.button('Predict')
    with col2:
        me2=st.button('Head to Head')
    
    if(me):
        if(k!=j):
            pretoss(k,j)
            
            #pretoss(k,j)
        else:
            st.error("Team Names should be different")
    if(me2):
        matches = pd.read_csv('matches.csv')
        
        x=['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
            'Rising Pune Supergiant', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
            'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
            'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants', 'Delhi Capitals']
        #y = [1,2,3,4,5,6,7,8,9,10,11,12,13,4,14]
        y = ['SRH','MI','GL','RPS','RCB','KKR','DC','KXIP','CSK','RR','SRH1','KTK','PW','RPS','DC']
        
        matches.replace(x,y,inplace = True)
        #matches
        
        lamela=matches[((matches['team1']==k)|(matches['team1']==j))&((matches['team2']==k)|(matches['team2']==j))]
        lamela['winner'].value_counts()
        
        aaaa=list(lamela['winner'].value_counts().values)
        bbbb=list(lamela['winner'].value_counts().index)
        
        team_colors={'RCB':'Red','MI':'Blue','DC':'royalblue','KKR':'rebeccapurple','SRH':'Orange','RR':'violet','CSK':'Yellow','KXIP':'orangered'}
        #colors = ['gold', 'mediumturquoise']
        #colors = [ 'darkorange', 'lightgreen']
        team_c=[]
        team_n=[]
        for i,j in team_colors.items():
            for k in bbbb:
                if(k==i):
                    team_c.append(j)
                    team_n.append(i)
        #print(team_n[0])
        if((team_n[0]==bbbb[0])):
            colors=team_c
            #print('h')
        else:
            team_c.reverse()
            colors=team_c
        fig = go.Figure(data=[go.Pie(labels=bbbb, values=aaaa,hole=0.1,pull=[0, 0.1])])
        fig.update_layout(height=600,width=600, title_text="Head to Head "+bbbb[0]+'  vs '+bbbb[1], font=dict(family='Courier New, monospace', size=20, color='#000000'))
        fig.update_traces(hoverinfo='label+percent', textinfo='value+label', textfont_size=18,marker=dict(colors=colors, line=dict(color='#000000', width=2)))
        st.plotly_chart(fig,use_container_width=True)
        st.write("### The above Pie Chart represents head to head encounter between "+bbbb[0]+" & "+bbbb[1])
    
    
elif(q=='Post-Toss Prediction 🏏'):
    st.title('Post-Toss Prediction')
    k=st.selectbox("Team 1",['RCB', 'MI', 'SRH', 'CSK', 'KKR', 'DC', 'RR', 'KXIP'])
    j=st.selectbox("Team 2",['MI', 'SRH', 'CSK', 'KKR', 'DC', 'RR', 'KXIP', 'RCB'])
    if(k==j):
        st.error("Team Names should be different")
    st.write("**Note**:  "+k+" vs "+j+" is not same as "+j+" vs "+k)
    l=st.radio("Toss winner",[k,j])
    bat='Bat'
    field='Field'
    m=st.radio("Toss Decision",[bat,field])
    st.subheader(k+" vs "+j)
    st.subheader(l+" won the toss and opted to "+m)
    st.write()
    
    #me=st.button("Predict")
    
    me, col2 = st.beta_columns([1,0.225])
    
    with me:
        me=st.button('Predict')
    with col2:
        me2=st.button('Head to Head')

    if(me):
        if(k!=j):
            posttoss(k,j,l,m.lower())
        else:
            st.error("Team Names should be different")
    
    if(me2):
        matches = pd.read_csv('matches.csv')
        
        x=['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
            'Rising Pune Supergiant', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
            'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
            'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants', 'Delhi Capitals']
        #y = [1,2,3,4,5,6,7,8,9,10,11,12,13,4,14]
        y = ['SRH','MI','GL','RPS','RCB','KKR','DC','KXIP','CSK','RR','SRH1','KTK','PW','RPS','DC']
        
        matches.replace(x,y,inplace = True)
        #matches
        
        lamela=matches[((matches['team1']==k)|(matches['team1']==j))&((matches['team2']==k)|(matches['team2']==j))]
        lamela['winner'].value_counts()
        
        aaaa=list(lamela['winner'].value_counts().values)
        bbbb=list(lamela['winner'].value_counts().index)
        
        team_colors={'RCB':'Red','MI':'Blue','DC':'royalblue','KKR':'rebeccapurple','SRH':'Orange','RR':'violet','CSK':'Yellow','KXIP':'orangered'}
        #colors = ['gold', 'mediumturquoise']
        #colors = [ 'darkorange', 'lightgreen']
        team_c=[]
        team_n=[]
        for i,j in team_colors.items():
            for k in bbbb:
                if(k==i):
                    team_c.append(j)
                    team_n.append(i)
        #print(team_n[0])
        if((team_n[0]==bbbb[0])):
            colors=team_c
            #print('h')
        else:
            team_c.reverse()
            colors=team_c
        fig = go.Figure(data=[go.Pie(labels=bbbb, values=aaaa,hole=0.1,pull=[0, 0.1])])
        fig.update_layout(height=600,width=600, title_text="Head to Head "+bbbb[0]+'  vs '+bbbb[1], font=dict(family='Courier New, monospace', size=20, color='#000000'))
        fig.update_traces(hoverinfo='label+percent', textinfo='value+label', textfont_size=18,marker=dict(colors=colors, line=dict(color='#000000', width=2)))
        st.plotly_chart(fig,use_container_width=True)
        st.write("### The above Pie Chart represents head to head encounter between "+bbbb[0]+" & "+bbbb[1])


elif(q=='IPL Data Analytics 📈'):
    #st.title('This area is under construction')
    allballs = pd.read_csv('all_deliveries.csv')
    matches = pd.read_csv('matches.csv')
    x=['Rising Pune Supergiant']
    y=['Rising Pune Supergiants']
    allballs.replace(x,y,inplace = True)
    
    o=st.selectbox("Select Stats",['Select','Team Stats','Player Stats'])
    
    if(o=='Select'):
        st.write("Select Stat type")
        st.write ("""
              - Team Stats
              - Player Stats
               """
                )
    
    elif(o=='Team Stats'):
        #st.write("bye")
        
        st.write("## Team wise performance")
        
        barWidth = 0.30
        # set height of bar
        bars1 = [203, 199, 195, 194, 192, 190, 178, 161, 46, 30, 30, 14]
        bars2 = [120, 106, 99, 95, 91, 88, 86, 81, 15, 13, 12, 6]
        bars3 = [59.09, 53.30, 50.8, 49.0, 47.4, 46.30, 48.3, 50.3, 32.6, 43.3, 40.0, 42.9]
        bars4 = ['MI', 'SRH', 'RCB', 'KKR', 'DC', 'KXIP', 'CSK', 'RR', 'PW', 'RPS', 'GL', 'KTK']
        # Set position of bar on X axis
        r1 = np.arange(len(bars1))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        # Make the plot
        fig,ax=plt.subplots()
        plt.bar(r1, bars1,  color='blue', width=barWidth, label='Played')
        plt.bar(r2, bars2, color='red', width=barWidth, label='Won')
        plt.bar(r3, bars3, color='orange', width=barWidth, label='Win Percentage ')
        # Add xticks on the middle of the group bars
        plt.xlabel('Teams', fontweight='bold')
        plt.ylabel('Matches', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(bars1))], ['MI', 'SRH', 'RCB', 'KKR', 'DC', 'KXIP', 'CSK', 'RR', 'PW', 'RPS', 'GL', 'KTK'])
        #to remove grid
        plt.grid(False) 
        # Create legend & Show graphic
        plt.legend()
        st.pyplot(fig)
        st.write("### The above Bar chart represents, Team wise analysis on Total Matches Played, Total Matches Won and Win percentage in IPL since 2008")
        
        st.write("  ")
        st.write("  ")
                
        st.write("## Most Man of the Match Awards")
        
        aa=list(matches['player_of_match'].value_counts().values)
        bb=list(matches['player_of_match'].value_counts().index)
        fig = px.bar(y=aa[:15],x=bb[:15],color=bb[:15],width=1000)
        fig.update_layout(xaxis_title="Players",yaxis_title="Count",
            font=dict(family="Courier New, monospace",size=18,color="RebeccaPurple"))
        st.plotly_chart(fig,use_container_width=True) 
        st.write("### The above Bar chart represents, Players who recieved most Man of the match awards in IPL since 2008")
        
        st.write("  ")
        st.write("  ")
        
        st.write("## Toss winners' precentage of Winning the match")
        figx,ax=plt.subplots()
        winnerper= matches[matches['winner']==matches['toss_winner']]
        k=len(winnerper)/matches['season'].value_counts().sum()
        plt.pie([k,1-k],explode=(0.05,0),shadow=True, startangle=90,labels=['Yes','No'],autopct='%1.1f%%')
        #plt.title("Toss winners' precentage of Winning the match",loc='center',fontweight='bold')
        plt.axis('equal')
        st.pyplot(figx)
        st.write("### The above Pie chart represents, Teams that win the toss has "+str(round((k*100),2))+"% record in winning the match in IPL since 2008")
        
        st.write("  ")
        st.write("  ")
        
        st.write("## Toss decision imapact on Winning the match")
        figy,ax=plt.subplots()
        h=matches['toss_decision']=='bat' #293 opted to bat)
        j=(matches['toss_winner']==matches['winner']).sum() #393(toss,match =win)
        # Teams that won toss and choose to Bat and went on to win the matches
        p=((matches['toss_decision']=='bat') & (matches['winner']==matches['toss_winner'])).sum() #134(both)
        q=p/j
        plt.pie([q,1-q],explode=(0.05,0),shadow=True, startangle=90,labels=['Bat First','Field First'],autopct='%1.1f%%')
        #plt.title("Toss decision imapact on Winning the match",loc='center',fontweight='bold')
        st.pyplot(figy)
        st.write("### The above Pie chart represents, Teams that win the toss and elect to Bat has "+str(round((q*100),1))+"% record in winning the match where as Teams that win the toss and elect to Field has "+str(100-round((q*100),1))+"% record in winning the match in IPL since 2008")

             
                
    elif(o=='Player Stats'):
        
        f=st.selectbox("Player Type",['Select','Batsman Stats','Bowler Stats'])
        
        if(f=='Select'):
            st.write("Select Player type")
            st.write ("""
              - Batsman Stats
              - Bowler Stats
               """
                )
        
        elif(f=='Batsman Stats'):
            nm=['Player','V Kohli', 'SK Raina', 'DA Warner', 'RG Sharma', 'S Dhawan', 'AB de Villiers', 'CH Gayle', 'MS Dhoni', 'RV Uthappa', 
            'G Gambhir', 'AM Rahane', 'SR Watson', 'KD Karthik', 'AT Rayudu', 'MK Pandey', 'YK Pathan', 'KA Pollard', 'BB McCullum', 
            'PA Patel', 'Yuvraj Singh', 'V Sehwag', 'KL Rahul', 'M Vijay', 'SV Samson', 'SE Marsh', 'JH Kallis', 'DR Smith', 'SR Tendulkar', 
            'SPD Smith', 'F du Plessis', 'SS Iyer', 'R Dravid', 'RA Jadeja', 'RR Pant', 'AC Gilchrist', 'JP Duminy', 'SA Yadav', 'AJ Finch', 'WP Saha', 'MEK Hussey']
            ssn=['All',2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,2021]
            
            g=st.selectbox("Select Batsman",nm)
            #zxc=st.slider("Season",min_value=2008, max_value=2020, step=1)
            
            if(g=='Player'):
                st.markdown("## Select Batsman")
            else:
                zx=st.select_slider("Season",options=ssn)
                st.write("### ☝️ Move the Slide bar (towards 👉) to get season specific stats (All-seasons are covered by default)")
                st.write(" ")
                st.write("  ")
                if(zx=='All'):
                    #st.write("under cover")
                    player_name=g
                    kj=allballs[(allballs['striker']==player_name)  & (allballs['innings']<3)]
                    
                    #dhh = pd.DataFrame({' Run type': 'ones dots fours twos sixes threes'.split(), 'Value': kj['runs_off_bat'].value_counts().values})
                    #st.write(dhh)
                    
                    total=kj['runs_off_bat'].sum()
                    #st.write("Total runs: ",total)
                    
                    #try_dff=['Total runs','Innings Played', 'Balls Faced','Ones','Twos','Threes','Fours','Sixes','Strike Rate','Average']
                    
                    inn=allballs[(allballs['striker']==player_name)|(allballs['non_striker']==player_name)]
                    mp=len(inn["match_id"].unique())
                    #st.write("Innings Played: ", mp)
                    
                    bfwe=len(allballs[(allballs['striker']==player_name)  & (allballs['innings']<3) & (allballs['wides']>0)])
                    bf=len(allballs[(allballs['striker']==player_name) & (allballs['innings']<3)])
                    #st.write("Balls Faced :",bf-bfwe)
                    
                    ones=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==1) & (allballs['innings']<3)])
                    #st.write("ones: ",ones)
                    
                    twos=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==2) & (allballs['innings']<3)])
                    #st.write("twos: ",twos)
                    
                    threes=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==3) & (allballs['innings']<3)])
                    #st.write("threes: ",threes)
                    
                    fours=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==4) & (allballs['innings']<3)])
                    #st.write("fours: ",fours)
                    
                    sixes=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==6) & (allballs['innings']<3)])
                    #st.write("sixes: ",sixes)
                    
                    #print("SR: ",(total/(bf-bfwe))*100)
                    #st.write("SR: ",(total/(bf-bfwe))*100)
                    
                    out=len(allballs[((allballs['striker']==player_name)|(allballs['non_striker']==player_name))  & (allballs['player_dismissed']==player_name) & (allballs['innings']<3)])
                    #print("Dismissed: ",out)
                    
                    #st.write("Avg: ",total/(out))
                    
                    #print(try_df)
                    #print(try_dff)
                    
                    #st.write(try_df)
                    #st.write(try_dff)
                    #st.table(out)
                    st.write("### IPL Career summary of "+g)
                    lk2={'Total runs': total, 'Innings Played': mp, 'Balls Faced':bf-bfwe, 'Ones':ones, 'Twos':twos, 'Threes':threes, 'Fours':fours, 'Sixes':sixes, 'Strike Rate':((total/(bf-bfwe))*100), 'Average':(total/(out))}
                    dcv=pd.DataFrame(lk2,index=[0],dtype=float)
                    st.table(dcv)
                    
                    #st.dataframe(dcv)
                    
                    st.write("  ")
                    
                    ssn1=[2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,2021]
                    scli=[]
                    for i in ssn1:
                        kj1=allballs[(allballs['striker']==g)  & (allballs['innings']<3) & (allballs['season']==i)]
                        scli.append(kj1['runs_off_bat'].sum())
                    #print(scli)
                    
                    fig = go.Figure(data=go.Scatter(x=ssn1, y=scli,line_color='rgb(0,100,80)'))
                    fig.update_layout(title="Season-wise Runs of "+g,xaxis_title="Season",yaxis_title="No of Runs", font=dict(family="Courier New, monospace",size=18,color="black"))
                    st.plotly_chart(fig,use_container_width=True)
                    st.write("### The above line-chart represents season-wise runs scored by "+g)
                    
                    st.write("  ")
                    st.write("  ")
                    
                    fig = go.Figure(data=[go.Pie(labels=['Ones','Twos','Threes','Fours','Sixes'], textinfo='label+percent',values=[ones,twos,threes,fours,sixes], hole=.2)])
                    fig.update_layout(height=600, title_text=player_name+"'s Runs scored", font=dict(family='Courier New, monospace', size=18, color='#000000'))
                    st.plotly_chart(fig,use_container_width=True)
                    st.write("### The above pie-chart represents percentage of runs scored by "+g)
                else:
                    player_name=g
                    season_yr=zx
                    kj=allballs[(allballs['striker']==player_name) & (allballs['season']==season_yr) & (allballs['innings']<3)]
                    total=kj['runs_off_bat'].sum()
                    #print("Total runs: ",total)
                    
                    #st.write("Total runs: ",total)
                    
                    
                    bfwe=len(allballs[(allballs['striker']==player_name) & (allballs['season']==season_yr) & (allballs['innings']<3) & (allballs['wides']>0)])
                    bf=len(allballs[(allballs['striker']==player_name) & (allballs['season']==season_yr) & (allballs['innings']<3)])
                    #print("BF :",bf-bfwe)
                    
                    ones=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==1)& (allballs['season']==season_yr) & (allballs['innings']<3)])
                    #print("ones: ",ones)
                    
                    twos=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==2)& (allballs['season']==season_yr) & (allballs['innings']<3)])
                    #print("twos: ",twos)
                    
                    threes=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==3)& (allballs['season']==season_yr) & (allballs['innings']<3)])
                    #print("threes: ",threes)
                    
                    fours=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==4)& (allballs['season']==season_yr) & (allballs['innings']<3)])
                    #print("fours: ",fours)
                    
                    sixes=len(allballs[(allballs['striker']==player_name) & (allballs['runs_off_bat']==6)& (allballs['season']==season_yr) & (allballs['innings']<3)])
                    #print("sixes: ",sixes)
                    
                    #rint("SR: ",(total/(bf-bfwe))*100)
                    
                    inn=allballs[(allballs['striker']==player_name) & (allballs['season']==season_yr)]
                    mp=len(inn["match_id"].unique())
                    #print("Matches Played: ", mp)
                    
                    out=len(allballs[(allballs['striker']==player_name) & (allballs['season']==season_yr) & (allballs['player_dismissed']==player_name) & (allballs['innings']<3)])
                    #print("Dismissed: ",out)
                    
                    #print("Average: ",total/(out))
                    st.write("### IPL - "+str(zx)+" summary of "+g)
                    lk2={'Total runs': total, 'Innings Played': mp, 'Balls Faced':bf-bfwe, 'Ones':ones, 'Twos':twos, 'Threes':threes, 'Fours':fours, 'Sixes':sixes, 'Strike Rate':((total/(bf-bfwe))*100), 'Average':(total/(out))}
                    dcv1=pd.DataFrame(lk2,index=[0],dtype=float)
                    st.table(dcv1)
                    #st.dataframe(dcv1)
                    
                    st.write("  ")
                    
                    fig = go.Figure(data=[go.Pie(labels=['Ones','Twos','Threes','Fours','Sixes'], textinfo='label+percent',values=[ones,twos,threes,fours,sixes], hole=.2)])
                    fig.update_layout(height=600, title_text=player_name+"'s Runs scored", font=dict(family='Courier New, monospace', size=18, color='#000000'))
                    st.plotly_chart(fig,use_container_width=True)
                    st.write("### The above pie-chart represents percentage of runs scored by "+g+" in "+str(zx))
                    
                    
        
        elif(f=='Bowler Stats'):
            #st.write("Bowlers")
            nmm=['Player','SL Malinga', 'A Mishra', 'PP Chawla', 'DJ Bravo', 'Harbhajan Singh', 'R Ashwin', 'B Kumar', 'SP Narine', 'YS Chahal', 'UT Yadav', 'RA Jadeja', 'JJ Bumrah',
                 'Sandeep Sharma', 'A Nehra', 'R Vinay Kumar', 'Z Khan', 'DW Steyn', 'SR Watson', 'MM Sharma', 'P Kumar', 'RP Singh', 'PP Ojha', 'DS Kulkarni', 'JA Morkel', 
                 'JD Unadkat', 'IK Pathan', 'CH Morris', 'AR Patel', 'Imran Tahir', 'M Morkel', 'L Balaji', 'Rashid Khan', 'MM Patel', 'I Sharma', 'R Bhatia', 'MJ McClenaghan',
                 'AB Dinda', 'JH Kallis', 'SK Trivedi', 'M Muralitharan', 'TA Boult', 'AD Russell', 'MG Johnson', 'K Rabada', 'KA Pollard', 'Mohammed Shami', 'Shakib Al Hasan', 
                 'JP Faulkner', 'KV Sharma', 'SK Warne', 'S Kaul']
            g=st.selectbox("Select Bowler",nmm)
            
            if(g=='Player'):
                st.markdown("## Select Bowler")
            else:
                mnb=allballs[(allballs['bowler']==g) & (allballs['innings']<3) &(allballs['noballs']!=1) & (allballs['wicket_type']!="run out")]
                #mnb
                
                wiccn=[]
                ssn1=[2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,2021]
                for i in ssn1:
                    mnb1=allballs[(allballs['bowler']==g) & (allballs['innings']<3) &(allballs['noballs']!=1) & (allballs['wicket_type']!="run out") & (allballs['season']==i)]
                    c=mnb1['wicket_type'].value_counts().values.sum()
                    wiccn.append(c)
                #print(wiccn)
                fig = go.Figure(data=go.Scatter(x=ssn1, y=wiccn,line_color='rgb(0,100,80)'))
                fig.update_layout(title="Wickets slpit of "+g,xaxis_title="Season",yaxis_title="No of Runs", font=dict(family="Courier New, monospace",size=18,color="black"))
                st.plotly_chart(fig,use_container_width=True)
                st.write("### The above line-chart represents season-wise wickets taken by "+g)
                
                st.write(" ")
                st.write(" ")
                st.dataframe(mnb['wicket_type'].value_counts())
                
                st.write("Total wickets taken by "+g,sum(mnb['wicket_type'].value_counts().values))
                st.write(" ")
                st.write(" ")
                
                aa=list(mnb['wicket_type'].value_counts().values)
                bb=list(mnb['wicket_type'].value_counts().index)
                fig = px.bar(y=aa,x=bb,color=bb)
                fig.update_layout(title="Wickets slpit of "+g,xaxis_title="Types of Dismissals",yaxis_title="No of dismissals",
                    font=dict(family="Courier New, monospace",size=14,color="RebeccaPurple"))
                st.plotly_chart(fig,use_container_width=True)
                st.write("### The above Bar-graph represents total wickets split by "+g)
            
    #st.balloons()
else:
    
    #st.write('Should write description')
    #st.write( """welcome
              
    
    with st.beta_expander("⚙️  Usage Heirarchy ", expanded=False):
        st.write ("""
            - ### IPL Data Analytics 📈:
               > - Team Stats
               > - Player Stats:
                   - Batsman Stats
                   - Bowler Stats
               
            - ### Pre Toss Prediction 🏏
                > - In this section the ML model will predict Pre-Toss Sims between the selected teams
                    
            - ### Post Toss Prediction 🏏
                > - In this section the ML model will predict Post-Toss Sims between the selected teams"""
                )
    st.markdown("---")
    st.write("**Note**: *This project is only for entertainment purpose D0-NOT invest anything anywhere based upon these predictions (you never know) * ")
    #st.markdown("---")
    st.markdown("Developed by [Ravi Teja Rajavarapu](mailto:ravitejarajavarapu515@gmail.com) & [Pratik Satpati](mailto:pratiksatpati2013@gmail.com), feedback and suggestions are welcomed 📧")
    #st.markdown("---")
                
    st.balloons()
