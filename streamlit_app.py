import requests
import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from geopy.geocoders import Nominatim
import pydeck as pdk
from PIL import Image
import graphviz
from streamlit_agraph import agraph, Node, Edge, Config
from pydeck.types import String
from matplotlib_venn import venn2, venn3
import altair as alt
import plotly.graph_objects as go
import streamlit.components.v1 as components
#from chapter2 import chapter2

#st.set_option('deprecation.showPyplotGlobalUse', False)

import plotly.io as pio
pio.templates.default = "plotly"

st.set_page_config(page_title="CRNY Data Visualization", layout="wide")

# Custom CSS for horizontal radio buttons
horizontal_radio_css = """
    div[data-baseweb="radio"] > label {
        display: inline-block;
        margin-right: 20px;
    }
"""

# Custom CSS for text justification
text_justification_css = """
    p {
        text-align: justify;
    }
"""

# Combined custom CSS
mystyle = f'''
    <style>
        {horizontal_radio_css}
        {text_justification_css}
    </style>
'''

st.markdown(mystyle, unsafe_allow_html=True) 


with st.container():
    st.write("\n")

    # Setting the title of the app
    st.title('Economic Empowerment Through Art: A Journey of 2400 Artists')
    image_ny_artists_1 = Image.open('./assets/1.webp')
    
    new_width = 1200
    new_height = 500  # Specify the desired height
    image_ny_artists_1_resize = image_ny_artists_1.resize((new_width, new_height))
    #width=1750 use_column_width="auto"
    st.image(image_ny_artists_1_resize, width=None)

    # Introduction
    st.write("""
    Welcome to an illuminating exploration of the "Guaranteed Income for Artists" program by CRNY, a pioneering initiative that champions the economic empowerment of artists across New York State. This visionary program bestows 2,400 artists with a financial lifeline, offering \$1\,000 monthly payments over a span of 18 months, translating into a substantial $$43.2M investment in the arts. Our project unfurls the diverse tapestry of these artists' experiences, presenting a visual narrative that captures the profound impact of financial support on their well-being and creative output.
    """)

    # What to Expect
    st.header('What to Expect:')
    st.write("""
    - **Demographics Dive:** Witness the diversity of our artist community through vibrant visualizations that bring to life their varied backgrounds, from age and gender to artistic disciplines.
    - **Financial Fortitude:** Gain insights into the crucial role of financial stability in nurturing medical security, debt management, and the alleviation of economic uncertainties for our artists.
    - **Wellness Wonders:** Explore the remarkable connection between financial aid and improved wellness metrics, with a special focus on mental health, housing stability, and the empowering sense of purpose in life.
    - **Community and Connectivity:** Delve into the social fabric of the artistic community, understanding how financial independence enhances relationships, optimism, and a forward-looking mindset.
    - **The Gender Narrative:** Discover how the program fosters an inclusive space that respects and recognizes a spectrum of gender identities, ensuring equitable support for all.
    - **Program Impact and Future Horizons:** Reflect on the overarching success of the Guaranteed Income for Artists program, contemplating the potential for its replication and expansion to fortify the cultural bedrock of society.
    """)

    st.write("""
    This analysis goes beyond numbers; it's a story of transformation and inspiration. By bridging financial gaps, CRNY has not only fueled the artists' creative passions but has also reaffirmed the pivotal role of artists as the soul of our cultural heritage. We invite you to immerse yourself in this journey of change and to envision a future where financial security and artistic expression walk hand in hand.
    """)


with st.container():
    st.write("\n")
    st.title("The Artists of New York - Who Are They?")
    #st.markdown('<span style="font-size:18px; font-style: italic; font-family: \'Times New Roman\', Times, serif;"> Age distribution as per community </span>', unsafe_allow_html=True)


# Dataframe reading and pre-processing
download_link = 'https://drive.google.com/file/d/1_0bQfQQLhOGLLUQqBbx9NkyO-ihLSSuz/view?usp=drive_link'
df = pd.read_csv('gi_and_poa_survey_data.csv')

# Ethnicity
ethnicity_data = df['p38_race1']
ethnicity_counts_df = ethnicity_data.value_counts()
ethnicity_categories = ethnicity_counts_df.index.tolist()
ethnicity_counts = ethnicity_counts_df.values.tolist()

ethnicity_data_list = {
    'Ethnicity': ["White", "Black/African-American","Hispanic/Latin-American", "Asian", "No answer", "Other", "Arab or Middle Eastern", "Indigenous American", "Pacific Islander"],
    'Count': ethnicity_counts
}

# Gender
gender_data = df['p41_gender1']
gender_counts_df = gender_data.value_counts()
gender_labels = gender_counts_df.index.tolist()
gender_counts = gender_counts_df.values.tolist()

gender_data_list = {
    'Gender': ["Woman", "Man", "Non-binary", "No answer", "Other", "Twospirit"],
    'Count': gender_counts
}

# Community
community_data = df['p36_community']
community_counts_df = community_data.value_counts()
community_labels = community_counts_df.index.tolist()
community_counts = community_counts_df.values.tolist()

community_data_list = {
    'Community': community_labels,
    'Count': community_counts
}


# Age Range
age_range_data = df['p_agerange']
age_range_counts_df = age_range_data.value_counts()
age_range_labels = age_range_counts_df.index.tolist()
age_range_counts = age_range_counts_df.values.tolist()

age_range_data_list = {
    'Age Range': age_range_labels,
    'Count': age_range_counts
}


# LGBTQIAP
lgbtqiap_data = df['p43_lgbtqiap']
lgbtqiap_counts_df = lgbtqiap_data.value_counts()
lgbtqiap_labels = lgbtqiap_counts_df.index.tolist()
lgbtqiap_counts = lgbtqiap_counts_df.values.tolist()

lgbtqiap_data_list = {
    'LGBTQIAP': lgbtqiap_labels,
    'Count': lgbtqiap_counts
}


# Language
language_data = df['p40_language']
language_counts_df = language_data.value_counts()
language_labels = language_counts_df.index.tolist()
language_counts = language_counts_df.values.tolist()

language_data_list = {
    'Language': language_labels,
    'Count': language_counts
}

language_data_list['Language'][2] = "Other"
language_data_list['Language'][11] = "No answer"
language_data_list['Count'][2] = 377

del language_data_list['Language'][12]
del language_data_list['Count'][12]

language_df = pd.DataFrame(language_data_list)

fig_6 = px.bar(language_df, x='Language', y='Count', text='Count', color_discrete_sequence=["Magenta"])
fig_6.update_layout(width=500, height=500, yaxis_type='log', legend=dict(font=dict(size=10)))
fig_6.update_xaxes(title_text='', showticklabels=True)

image_ny_artists_1 = Image.open('./assets/diversity.png')
#width=1750 use_column_width="auto"
st.image(image_ny_artists_1, width=None)
#st.markdown('<span style="font-size:30px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Diversity of Guaranteed Income for Artists Participants </span>', unsafe_allow_html=True)
st.write("""
1) **Inclusivity in Support:** The majority representation from the LGBTQIA+ community reflects the program's commitment to inclusivity, providing support to historically marginalized groups.

2) **Support for Gender Diversity:** The significant number of transgender, nonbinary, and gender-nonconforming participants indicates the grant's reach within these communities, highlighting its role in fostering a gender-diverse artistic environment.

3) **Recognition of Caregivers:** The grant acknowledges the often-overlooked contributions of caregivers in the artistic community, providing financial assistance that can help them balance their creative work with caregiving responsibilities.

4) **Backing Immigrant Artists:** The program's support for immigrant artists showcases a recognition of the unique challenges they face, such as xenophobia and the threat of deportation, affirming the grant's role in empowering artists regardless of nationality or immigration status.

5) **Supporting Artists with Disabilities and Legal Involvement:** By assisting deaf, disabled, and legally involved individuals, the grant not only addresses financial stability but also signals a broader social impact by aiding those who may face significant barriers to full participation in the arts sector.
""")

st.markdown('<span style="font-size:30px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Gender Diversity Among Guaranteed Income for Artists Participant </span>', unsafe_allow_html=True)
image_ny_artists_1 = Image.open('./assets/gender_new.png')
#width=1750 use_column_width="auto"
st.image(image_ny_artists_1, width=None)
st.write("""
1) **Gender Representation:** The visualization reflects a progressive representation of gender identities, with women being the largest group. This could indicate either a greater number of women in the arts sector or a higher engagement rate with the grant program.

2) **Inclusivity of Gender Identities:** The notable presence of non-binary and other gender identities emphasizes the program's inclusivity, ensuring that support is not limited by traditional gender norms.

3) **Recognition of Indigenous Identities:** The specific inclusion and representation of Two-Spirit individuals acknowledge the importance of indigenous gender identities and the program's sensitivity to a wide spectrum of cultural identities.

4) **Opportunities for Engagement Improvement:** The 'not answered' category suggests that there is room for the program to encourage complete self-identification, which could help in tailoring support and resources even more effectively.

5) **Potential Outreach Initiatives:** Insights from the gender distribution can guide future outreach initiatives to ensure equitable access and support for all gender identities within the artistic community.
""")

# Your iframe embed code here (make sure to use the correct src URL)
iframe_embed_code = """
<iframe src="https://public.flourish.studio/visualisation/17042593/embed" frameborder="0" scrolling="no" style="width:100%; height:600px;"></iframe>
"""

# Embed the iframe in your Streamlit app
components.html(iframe_embed_code, height=600)

with st.container():
    #st.markdown('<span style="font-size:30px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Community </span>', unsafe_allow_html=True)
    st.write("Approximately 71% of New York artists reside in urban regions, a reflection of the state's overall development. Suburban environments that combine elements of urban and residential life are preferred by about 15% of people. A lesser but significant portion, 13%, comes from rural areas. These figures highlight the wide range of geographic origins of artists and highlight the pervasive impact of urban development on the formation of culture. This diverse residential landscape adds to New York's rich artistic fabric by reflecting the dynamic decisions made by artists.")


    col_1, col_2 = st.columns([1, 1], gap="large")
    

    # Ethnicity
    with col_1:
        st.markdown('<span style="font-size:25px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Urban </span>', unsafe_allow_html=True)
        ethnicity_image = Image.open('./assets/ethnicity.jpg')
        #st.image(ethnicity_image, width=None)
        st.write("1714 artists reported living in urban communities")
        st.markdown('<span style="font-size:25px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Rural </span>', unsafe_allow_html=True)
        ethnicity_image = Image.open('./assets/ethnicity.jpg')
        st.write("304 artists reported living in rural communities")
        
    # Age Range
    with col_2:
        st.markdown('<span style="font-size:25px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Suburban </span>', unsafe_allow_html=True)
        ethnicity_image = Image.open('./assets/ethnicity.jpg')
        #st.image(ethnicity_image, width=None)
        st.write("360 artists reported living in suburban communities")
        st.markdown('<span style="font-size:25px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Tribal </span>', unsafe_allow_html=True)
        ethnicity_image = Image.open('./assets/ethnicity.jpg')
        #st.image(ethnicity_image, width=None)
        st.write("22 artists reported living in tribal communities")
        #st.plotly_chart(fig_1,use_container_width = True)
    

    
iframe_embed_code = """
<iframe src="https://public.flourish.studio/visualisation/17041604/embed" style="width:100%; height:1000px;"></iframe>
"""

# Embed the iframe in your Streamlit app
components.html(iframe_embed_code, height=1000)

st.markdown('<span style="font-size:30px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Artist Demographics and Geographic Disbursement in CRNY\'s Statewide Support Initiative </span>', unsafe_allow_html=True)

with st.container():
    st.write("\n")
    #st.markdown('<span style="font-size:18px; font-style: italic; font-family: \'Times New Roman\', Times, serif;"> Let\'s see where our artists are from. </span>', unsafe_allow_html=True)
    county_ny_image = Image.open('./assets/2.webp')
    #st.image(county_ny_image, width=100)
    # Resize the image
    new_width = 1200
    new_height = 500  # Specify the desired height
    county_ny_image_resized = county_ny_image.resize((new_width, new_height))

    # Display the resized image
    #st.image(county_ny_image_resized)

    st.write("With 4434 artists, Kings County, which is home to important cities like Brooklyn, leads the competition's dataset. This is a thriving hotspot. New York County and New York City come in second and third, respectively, with 2837 artists, indicating a substantial artistic presence. With 1618 artists, Queens County claims the third position. Remarkably, 62 counties in the state of New York are included, exhibiting a wide range of geographic backgrounds. Notably, the least represented counties in terms of artists are Schuyler, Genesee, Wayne, and Madison; these counties are mainly suburban and rural. The scatterplot below shows the nuanced population distribution of artists across the New York State Counties. It is accompanied by a deck chart and heatmap.")
# County
county_data = df['p34_county']
county_counts_df = county_data.value_counts()

county_counts_df.index = county_counts_df.index.str.split(' County').str[0]
county_counts_df = county_counts_df.sort_index()
#st.write(county_counts_df.sort_values(ascending=False))
county_counts_df = county_counts_df.groupby(county_counts_df.index).sum()

county_labels = county_counts_df.index.tolist()
county_counts = county_counts_df.values.tolist()

county_locator = Nominatim(user_agent="county_locator")

def get_coordinates(county_name):
    location = county_locator.geocode(county_name + ', New York, USA')
    if location:
        return location.latitude, location.longitude
    else:
        return None

county_lats = [42.6511674, 42.2446061, 40.8466508, 42.1455623, 42.2234823, 42.8093409, 42.2894671, 42.1384667, 42.4784565, 44.7278943, 42.2415027, 42.5842136, 42.194917, 41.7194303, 42.352098, 44.0638879, 44.5599139, 43.1061507, 43.0102726, 42.2628769, 43.6307863, 43.4911326, 44.059311, 43.1509069, 43.7344277, 42.7360902, 42.875882, 41.3304767, 42.8941269, 40.7352994, 40.7127281, 43.2042439, 43.2144051, 43.015598, 42.8580624, 41.3873306, 43.2244513, 43.4112973, 42.5984272, 41.426996, 40.7135078, 42.7091389, 42.7905911, 41.1519319, 43.0833231, 42.8142432, 42.5757217, 42.3903231, 42.7831619, 44.4973591, 42.2359045, 40.8832318, 41.7156311, 42.1333395, 42.118333, 41.8689317, 43.5018687, 43.2294536, 43.1500557, 41.1763139, 42.7039813, 42.6444444]
county_longs = [-73.754968, -78.0419281, -73.8785937, -75.8404114, -78.6477096, -76.5700777, -79.421728, -76.7725493, -75.6130279, -73.6686982, -73.6723456, -76.0704906, -75.0016302, -73.7516205, -79.322965, -73.7542043, -74.3273735, -74.4461771, -78.1780196, -74.0878112, -74.4659275, -74.9481252, -75.9995742, -73.8542895, -75.440289, -77.7781416, -75.6802581, -74.1866348, -74.4099745, -73.5615778, -74.0060152, -78.7676017, -75.4039155, -76.2257127, -77.295025, -74.2507287, -78.2272835, -76.1279841, -75.0142701, -73.760156, -73.8283132, -73.5107732, -77.5319396, -74.0357266, -73.8712155, -73.9395687, -74.4390277, -76.8691575, -76.8386051, -75.0657043, -77.3750862, -72.8578027, -74.7804323, -76.3309339, -75.249444, -74.2618518, -73.8164637, -73.4471343, -77.0377603, -73.7907554, -78.2415228, -77.112177]

min_count = min(county_counts)
max_count = max(county_counts)
normalized_counts = [(count - min_count) / (max_count - min_count) for count in county_counts]

county_map_df = pd.DataFrame({
    'County': county_labels,
    'latitude': county_lats,
    'longitude': county_longs,
    'Count': county_counts
})

# Create DataFrame
chart_data = pd.DataFrame({
    'county': county_labels,
    'lat': county_lats,
    'lon': county_longs,
    'count': county_counts
})

chart_data['log'] = np.log(chart_data['count'])

# Scatterplot proper

st.pydeck_chart(pdk.Deck(
            map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=41.730610,
            longitude=-76,
            zoom=6,
            pitch=40,
        ),
        layers=[
            pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            opacity=0.2,
            stroked=True,
            filled=True,
            radius_scale=3000,
            line_width_min_pixels=1,
            get_position='[lon, lat]',
            get_radius='log',
            get_fill_color=[255, 140, 0],
            get_line_color=[0, 0, 0],
           
                  ),
               ],
            ))
         
# with st.container():
    # col_1, col_2= st.columns([1, 1], gap="small")

    # with col_1:
            # #PyDeck with rising bars

            # st.pydeck_chart(pdk.Deck(
                # map_style=None,
            # initial_view_state=pdk.ViewState(
                # latitude=41.730610,
                # longitude=-76,
                # zoom=5,
                    # pitch=50,
            # ),
            # layers=[
                # pdk.Layer(
                # 'HexagonLayer',
                # data=chart_data,
                # opacity=0.1,
                # get_position='[lon, lat]',
                # radius=2000,
                # get_elevation_weight = 'log',
                # elevation_scale=200,
                # elevation_range=[0, 1000],
                # pickable=True,
                # extruded=True,
                # ),
                # pdk.Layer(
                    # 'ScatterplotLayer',
                    # data=chart_data,
                    # get_position='[lon, lat]',
                    # get_color='[255, 140, 0, 160]',
                    # get_radius = 3000,
                # ),
            # ],
        # ))
            
    # with col_2:
                # # heatmap

                # st.pydeck_chart(pdk.Deck(
                        # map_style=None,
                        # initial_view_state=pdk.ViewState(
                        # latitude=41.730610,
                        # longitude=-76,
                        # zoom=5,
                        # pitch=50,
                    # ),
                    # layers=[
                    # pdk.Layer(
                        # 'HeatmapLayer',
                        # data=chart_data,
                        # opacity=1,
                        # get_position='[lon, lat]',
                        # threshold=0.9,
                        # aggregation=String('MEAN'),
                        # get_weight = 'log',
                    # ),
                # ],
            # ))
                
st.write("------")
image_ny_artists_1 = Image.open('./assets/fund_allocation.png')
#width=1750 use_column_width="auto"
#st.image(image_ny_artists_1, width=None)
col1, col2, col3 = st.columns([1,2,1])
# Display the image in the middle column
with col2:
    st.image(image_ny_artists_1, use_column_width=True)
st.write("""
1. **New York City's Artistic Epicenter:** With 65% (1,556 artists) of participants based in New York City, the graph underscores the city's dominance in the state's artistic landscape, mirrored by a substantial proportion of funding.

2. **Statewide Representation:** The reach of CRNY's program is extensive, with participants hailing from 60 of New York State's 62 counties, showing a commitment to supporting artists statewide.

3. **Funding Disparity:** While the majority of artists are based in New York City, resulting in a higher funding concentration there, the graph reveals significant investment in other regions, albeit at a lesser scale, indicating a nuanced approach to funding allocation.

4. **Regional Artist Populations:** The visualization helps identify regions with smaller artist populations, which could be critical for targeted growth and support initiatives in the artistic communities.

5. **Funding Opportunities:** The visual disparity between the number of artists and funding in regions outside New York City suggests potential opportunities for increasing financial support to balance the distribution of resources and aid.
""")


# Your iframe embed code here (make sure to use the correct src URL)
iframe_embed_code = """
<iframe src="https://public.flourish.studio/visualisation/17041755/embed" frameborder="0" scrolling="no" style="width:100%; height:600px;"></iframe>
"""

# Embed the iframe in your Streamlit app
components.html(iframe_embed_code, height=600)
st.write("""
1. **Cultural Melting Pot:** The bubble chart vividly illustrates the linguistic diversity of New York's artists, with Spanish being the most prominently spoken language after English, indicating a strong Hispanic influence within the artistic community.
2. **Multilingual Richness:** With artists speaking 32 different languages, the chart showcases New York's position as a cultural melting pot where a multitude of linguistic groups contributes to art.
3. **Minority Languages Representation:** Smaller bubbles for languages like Korean, Arabic, and Haitian Creole highlight the inclusion of minority language speakers, pointing to a diverse cultural exchange and cross-pollination of ideas in the arts.
4. **'Other' Languages:** The sizable 'Other' category suggests a vast array of less common languages that are not individually highlighted but together represent a significant portion of the artists, emphasizing the depth of linguistic diversity.
5. **Cultural Perspectives and Inclusion:** This linguistic variety underscores the necessity for inclusive programming that takes into account the broad range of cultural perspectives, potentially enriching the community's artistic output.
""")    



image_ny_artists_1 = Image.open('./assets/Financial_stability.png')
#width=1750 use_column_width="auto"
st.image(image_ny_artists_1, width=None)
#st.markdown('<span style="font-size:30px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Financial Stability for Artists Receiving Payment </span>', unsafe_allow_html=True)
st.write("""
1) **Medical Vulnerability:** 41% of artists (972 individuals) reported being vulnerable to medical issues, indicating a significant need for healthcare support within the artist community.

2) **Lack of a Financial Safety Net:** A significant 62% (1,333 artists) reported having no financial safety net, highlighting the critical role of the grant in providing financial security.

3) **Debt:** 55% of artists (1,311 individuals) indicated carrying unmanageable debt, suggesting that many artists are in precarious financial situations that the grant could help alleviate.

4) **Income Instability:** 56% (1,333 artists) reported income instability, emphasizing the importance of the grant in creating a more stable financial environment for artists to thrive.

5) **Grant Impact:** The data points to a substantial portion of the artist population benefiting from the grant in fundamental ways, addressing key financial challenges that could otherwise hinder their creative endeavors.
""")
#st.write("In conclusion, our exploration into New York's artistic community has revealed a multifaceted landscape, rich in diversity and creativity. Creatives Rebuild New York (CRNY) has positively influenced New York’s artistic landscape, enhancing artists’ financial, health, and housing stability. Acknowledging the demographic diversity of applicants, CRNY tailors support to artists of various ages, races, ethnicities, genders, and LGBTQIAP+ identities, recognizing the unique challenges each group faces. However, the COVID-19 pandemic's impact, which led to job losses and financial instability, underscores the need for ongoing and expanded public awareness. Effective outreach is essential to ensure artists are fully aware of the resources available to them. Moving forward, CRNY's continued collaboration with communities will be key in amplifying its initiatives, providing artists not only with the support they need but also the platforms to advocate for sustained change in public policy and social support systems.")
    

image_ny_artists_1 = Image.open('./assets/arts.png')
#width=1750 use_column_width="auto"
st.image(image_ny_artists_1, width=None)

#st.write("In conclusion, our exploration into New York's artistic community has revealed a multifaceted landscape, rich in diversity and creativity. Creatives Rebuild New York (CRNY) has positively influenced New York’s artistic landscape, enhancing artists’ financial, health, and housing stability. Acknowledging the demographic diversity of applicants, CRNY tailors support to artists of various ages, races, ethnicities, genders, and LGBTQIAP+ identities, recognizing the unique challenges each group faces. However, the COVID-19 pandemic's impact, which led to job losses and financial instability, underscores the need for ongoing and expanded public awareness. Effective outreach is essential to ensure artists are fully aware of the resources available to them. Moving forward, CRNY's continued collaboration with communities will be key in amplifying its initiatives, providing artists not only with the support they need but also the platforms to advocate for sustained change in public policy and social support systems.")
#st.markdown('<span style="font-size:30px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Artistic Discipline Diversity Among Selected Artists </span>', unsafe_allow_html=True)
st.write("""
1) **Predominance of Visual Arts and Music:** The representation of artists in Visual Arts and Music suggests these disciplines are particularly vibrant or well-supported within the artistic community.

2) **Diversity in Disciplines:** The range of disciplines, including "Others", indicates a rich tapestry of endeavors being pursued, showcasing the grant program's reach across various creative expression.

3) **Theater and Film's Significant Presence:** Theater and Film also have substantial representation, highlighting the importance of performance and media arts as avenues for artistic expression.
""")

image_ny_artists_1 = Image.open('./assets/aging_new.png')
#width=1750 use_column_width="auto"
st.image(image_ny_artists_1, width=None)

#st.markdown('<span style="font-size:30px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Age Range of Guaranteed Income for Artists Participants </span>', unsafe_allow_html=True)
st.write("""
1) **Generational Reach:** The grant appears to primarily support mid-career artists, with those in the 25-34 age range being the most engaged, suggesting that the grant program is resonating with individuals who may be looking to establish or solidify their careers.

2) **Emerging Artists:** Younger artists (18-24) are also engaging with the program, which could indicate that the grant serves as an important stepping stone for emerging talent.

3) **Sustaining Established Artists:** The artists in older age brackets shows that the grant also aids in sustaining artists as they continue to contribute to their fields beyond the peak years of career activity.

4) **Long-Term Support Considerations:** The less significant engagement from artists 55 and older may prompt discussions on tailoring grant support for the needs of senior artists or enhancing outreach efforts to ensure the grant's benefits are fully utilized across age groups.

5) **Diversity in Artistic Evolution:** The varied age range participation reflects the grant's capacity to support artists at different stages of their artistic evolution, which is essential for a vibrant and diverse cultural ecosystem.
""")






#st.markdown('<span style="font-size:30px; font-style: Bold; font-family: \'Times New Roman\', Times, serif;"> Artist Wellness Metrics for all Artists who have applied for GI programme </span>', unsafe_allow_html=True)

#st.write("In conclusion, our exploration into New York's artistic community has revealed a multifaceted landscape, rich in diversity and creativity. Creatives Rebuild New York (CRNY) has positively influenced New York’s artistic landscape, enhancing artists’ financial, health, and housing stability. Acknowledging the demographic diversity of applicants, CRNY tailors support to artists of various ages, races, ethnicities, genders, and LGBTQIAP+ identities, recognizing the unique challenges each group faces. However, the COVID-19 pandemic's impact, which led to job losses and financial instability, underscores the need for ongoing and expanded public awareness. Effective outreach is essential to ensure artists are fully aware of the resources available to them. Moving forward, CRNY's continued collaboration with communities will be key in amplifying its initiatives, providing artists not only with the support they need but also the platforms to advocate for sustained change in public policy and social support systems.")
image_ny_artists_1 = Image.open('./assets/wellness.png')
#width=1750 use_column_width="auto"
st.image(image_ny_artists_1, width=None)
st.write("""
1) **Mental Health Priority:** High average ratings for mental health signify that artists prioritize mental well-being, possibly reflecting an awareness of its importance to creativity and overall life satisfaction.

2) **Stability in Housing and Agency:** Stable housing and a sense of agency over the future receive positive ratings, indicating that artists feel relatively secure in these aspects, which are crucial for a sustainable artistic practice.

3) **Basic Needs and Health:** While the ability to feed themselves and physical health have moderate ratings, these scores suggest room for improvement in ensuring all artists have their basic needs met and maintain good health.

4) **Optimism and Social Connections:** For optimism and social relationships show that while there's a sense of hope and community, enhancing these areas will contribute to the artists' well-being.

5) **Purposeful Life:** A purposeful life has a lower rating compared to other metrics, pointing to potential areas where artists might seek support, such as in finding direction and meaning in their careers.
""")
#st.write("In conclusion, our exploration into New York's artistic community has revealed a multifaceted landscape, rich in diversity and creativity. Creatives Rebuild New York (CRNY) has positively influenced New York’s artistic landscape, enhancing artists’ financial, health, and housing stability. Acknowledging the demographic diversity of applicants, CRNY tailors support to artists of various ages, races, ethnicities, genders, and LGBTQIAP+ identities, recognizing the unique challenges each group faces. However, the COVID-19 pandemic's impact, which led to job losses and financial instability, underscores the need for ongoing and expanded public awareness. Effective outreach is essential to ensure artists are fully aware of the resources available to them. Moving forward, CRNY's continued collaboration with communities will be key in amplifying its initiatives, providing artists not only with the support they need but also the platforms to advocate for sustained change in public policy and social support systems.")

#st.write("In conclusion, our exploration into New York's artistic community has revealed a multifaceted landscape, rich in diversity and creativity. Creatives Rebuild New York (CRNY) has positively influenced New York’s artistic landscape, enhancing artists’ financial, health, and housing stability. Acknowledging the demographic diversity of applicants, CRNY tailors support to artists of various ages, races, ethnicities, genders, and LGBTQIAP+ identities, recognizing the unique challenges each group faces. However, the COVID-19 pandemic's impact, which led to job losses and financial instability, underscores the need for ongoing and expanded public awareness. Effective outreach is essential to ensure artists are fully aware of the resources available to them. Moving forward, CRNY's continued collaboration with communities will be key in amplifying its initiatives, providing artists not only with the support they need but also the platforms to advocate for sustained change in public policy and social support systems.")
    
with st.container():
    #st.subheader("Chapter 6: Reflections and Future Pathways")
    #future_image = Image.open('./assets/future.jpg')
    #st.image(future_image, width=None)
    #st.write("\n")
    #st.write("In conclusion, our exploration into New York's artistic community has revealed a multifaceted landscape, rich in diversity and creativity. Creatives Rebuild New York (CRNY) has positively influenced New York’s artistic landscape, enhancing artists’ financial, health, and housing stability. Acknowledging the demographic diversity of applicants, CRNY tailors support to artists of various ages, races, ethnicities, genders, and LGBTQIAP+ identities, recognizing the unique challenges each group faces. However, the COVID-19 pandemic's impact, which led to job losses and financial instability, underscores the need for ongoing and expanded public awareness. Effective outreach is essential to ensure artists are fully aware of the resources available to them. Moving forward, CRNY's continued collaboration with communities will be key in amplifying its initiatives, providing artists not only with the support they need but also the platforms to advocate for sustained change in public policy and social support systems.")
    st.markdown('<span style="font-size:30px; font-style: italic; font-family: \'Times New Roman\', Times, serif;"> Thank You. </span>', unsafe_allow_html=True)
    st.markdown('<span style="font-size:15px; font-style: italic; font-family: \'Times New Roman\', Times, serif;"> by Sandeep Kumar, Ravi Teja Rajavarapu </span>', unsafe_allow_html=True)
