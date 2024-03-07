import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import random
import math




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

st.title('Creatives Rebuild New York (CRNY) Data Visualization')

st.markdown("***Study Done by:*** **Sandeep Kumar** & **Ravi Teja Rajavarapu**")

st.markdown("***Contact info*** [Sandeep Kumar](mailto:sk175@iu.edu) & [Ravi Teja Rajavarapu](mailto:rrajavar@iu.edu)")
    #st.markdown("---")

st.write("***College and Department info:*** ***Indiana Univerity Bloomington***, Luddy School of Informatics, Computing, and Engineering")

st.write("***Student Status:*** Graduate")
st.write("***Title:*** Creatives Rebuild New York (CRNY) Data Visualization ")


## Your iframe embed code here (make sure to use the correct src URL)
#iframe_embed_code = """
#<iframe src="https://public.flourish.studio/visualisation/17042593/embed" frameborder="0" scrolling="no" style="width:100%; height:600px;"></iframe>
#"""

# Embed the iframe in your Streamlit app
#components.html(iframe_embed_code, height=600)



# Read the CSV file into a DataFrame named "df5"
df5 = pd.read_csv("df5.csv")

# Show the column names of the DataFrame
df5.columns.tolist()

dictionary_ques = pd.read_excel('Book1.xlsx')

dictionary_ques.head()


# Convert the DataFrame to a dictionary
# Initialize an empty dictionary
data_dict = {}

# Initialize current_key to None
current_key = None

# Loop through each row in the DataFrame
for index, row in dictionary_ques.iterrows():
    question = row['questions']
    column_name = row['column_names']
    
    # Check if 'question' is NaN. If not, update current_key
    if pd.notna(question):
        current_key = question
        data_dict[current_key] = []
        
    # Append 'column_name' to the list corresponding to 'current_key'
    if pd.notna(column_name):
        data_dict[current_key].append(column_name)

print(data_dict)

# key_main = "artistic_approach"
# key_columns = data_dict[key_main]

# # Calculate the count of 1s in each column
# count_of_ones = df5[key_columns].sum()

# # Sort the columns by descending order of the count of 1s
# sorted_columns = count_of_ones.sort_values(ascending=False).index

# # Remove "approach_" from the x-axis parameter labels
# cleaned_labels = [label.replace("approach_", "") for label in sorted_columns]

# # Create a horizontal bar plot with fading colors
# plt.figure(figsize=(13, 6))
# palette = sns.color_palette("Blues_r", len(cleaned_labels))  # Fading colors from high to low
# ax = sns.barplot(x=count_of_ones[sorted_columns], y=cleaned_labels, orient="h", palette=palette)

# # Annotate the count of each category inside the bar
# for i, v in enumerate(count_of_ones[sorted_columns]):
#     ax.text(v + 5, i, str(v), va='center', fontsize=12)  # Adjust the position for better readability

# plt.title('Artistic Approach')  # Updated title
# plt.xlabel('Number of Artists →')
# plt.ylabel('Approach →')
# plt.show()


st.subheader("Artistic Practice and Approach")

st.write("The visualizations below provide insights into artists diverse approaches, ages, and artistic disciplines.")

# Your existing code for data preparation and plotting
def create_plot(data_dict,df5):
    key_main = "artistic_approach"
    key_columns = data_dict[key_main]

    # Calculate the count of 1s in each column
    count_of_ones = df5[key_columns].sum()

    # Sort the columns by descending order of the count of 1s
    sorted_columns = count_of_ones.sort_values(ascending=False).index

    # Remove "approach_" from the x-axis parameter labels
    cleaned_labels = [label.replace("approach_", "") for label in sorted_columns]

    # Create a horizontal bar plot with fading colors
    plt.figure(figsize=(15, 8))
    palette = sns.color_palette("Blues_r", len(cleaned_labels))  # Fading colors from high to low
    ax = sns.barplot(x=count_of_ones[sorted_columns], y=cleaned_labels, orient="h", palette=palette)

    # Annotate the count of each category inside the bar
    for i, v in enumerate(count_of_ones[sorted_columns]):
        ax.text(v + 5, i, str(v), va='center', fontsize=12)  # Adjust the position for better readability

    plt.title('Artistic Approach')
    plt.xlabel('Number of Artists →')
    plt.ylabel('Approach →')
    return plt

# Streamlit app
# st.title("Artistic Approach Visualization")

# Create the plot
fig = create_plot(data_dict,df5)

# Display the plot
st.pyplot(fig)
# st.write("This is new  one test")

st.write("The bar chart ""Artistic Approach"" shows that solo work is the most common approach among artists, followed by collaboration with other artists. To support this creative ecosystem, fostering environments that support individual artistic endeavors and collaborative opportunities could be key, particularly for those in less represented approaches like cross-sector collaboration.")
st.markdown("")
st.markdown("")
st.markdown("")


# st.pyplot(fig)
#         st.write("### The above Bar chart represents, Team wise analysis on Total Matches Played, Total Matches Won and Win percentage in IPL since 2008")


def create_plot_2(data_dict,df5):
    # Set style to 'white' which removes the grid
    sns.set(style="white")

    # Create the histogram
    plt.figure(figsize=(15, 8))
    n, bins, patches = plt.hist(df5['p_age'], bins=20, edgecolor='black')

    # Normalize the bin heights to get the frequency for color intensity
    bin_max = max(n)
    crimson_rgb = (220/255, 20/255, 60/255)  # RGB for crimson
    colors = [(*crimson_rgb, n_i/bin_max) for n_i in n]  # Fade effect with alpha value

    # Apply color gradient to each bar in the histogram
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    # Add titles and labels
    plt.title("Age Distribution of Artists", fontsize=18, fontweight='bold', color='darkslateblue')
    plt.xlabel("Age →", fontsize=14, fontweight='bold', color='darkslateblue')
    plt.ylabel("Count of Artists → ", fontsize=14, fontweight='bold', color='darkslateblue')

    # Calculate and display the average age on the right-hand side of the graph
    average_age = df5['p_age'].mean()
    plt.text(0.95, 0.95, f"Average Age: {average_age:.1f}", transform=plt.gca().transAxes, horizontalalignment='right',
             verticalalignment='top', fontsize=14, color='darkred', fontweight='bold')
    # Draw a vertical line at the average age
    plt.axvline(average_age, color='DarkGreen', linestyle='--', linewidth=2)

    # Customize ticks
    plt.xticks(fontsize=12, color='slategray')
    plt.yticks(fontsize=12, color='slategray')

    # Remove top and right spines for a cleaner look
    sns.despine()

    plt.tight_layout()
    # plt.show()
    return plt
fig = create_plot_2(data_dict,df5)

# Display the plot
st.pyplot(fig)

st.write("The histogram Age Distribution of Artists shows a concentration of artists in the mid-age ranges, with the highest number of artists in the 30-39 age bracket. The average age of artists in the dataset is 36.2 years. This indicates that the majority of artists are in their mid-career stage, which could inform support programs to be geared towards sustaining and developing artists who are likely established yet still evolving in their careers.")
st.markdown("")


st.markdown("***What artistic disciplines are most common among respondents?***")

# def create_plot_3(data_dict,df5):
#     # Assuming 'df5' is your dataframe and 'data_dict' contains the mapping for the 'art_discipline' column
#     key_main = "art_discipline"  # Replace with actual column name for artistic disciplines
#     key_columns = data_dict[key_main]  # Replace with actual column mappings

#     # Calculate the count of 1s in each column
#     count_of_ones = df5[key_columns].sum()

#     # Sort the counts in descending order to find the most common disciplines
#     sorted_counts = count_of_ones.sort_values(ascending=False)

#     # Remove 'discip_' from the column labels
#     sorted_counts.index = sorted_counts.index.str.replace('discip_', '')

#     # Calculate the percentage for each discipline
#     total_responses = len(df5)
#     percentage = (sorted_counts / total_responses) * 100

#     # Create a color gradient in the crimson color range with fade effect
#     n = len(sorted_counts)
#     crimson_color = (0.6, 0, 0)  # RGB for crimson color
#     colors = [(*crimson_color, i/n) for i in range(n, 0, -1)]  # Fade effect with alpha value

#     # Create a bar plot to visualize the counts and percentages
#     plt.figure(figsize=(15, 10))
#     ax = plt.bar(sorted_counts.index, sorted_counts.values, color=colors)

#     # Customize the plot for a more classy and artistic look
#     plt.title("Most Common Artistic Disciplines", fontsize=18, fontweight='bold', color='darkslategray')
#     plt.xlabel("Artistic Discipline", fontsize=14, fontweight='bold', color='darkslategray')
#     plt.ylabel("Count of Artists →", fontsize=14, fontweight='bold', color='darkslategray')
#     plt.xticks(rotation=90, ha="center", fontsize=11, color='black')
#     plt.yticks(color='slategray')

#     # Remove grid and set background color
#     plt.grid(False)
#     plt.gca().set_facecolor('whitesmoke')
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.gca().spines['left'].set_color('slategray')
#     plt.gca().spines['bottom'].set_color('slategray')

#     # Display the percentages on top of the bars
#     for rect, perc in zip(ax, percentage):
#         height = rect.get_height()
#         plt.text(rect.get_x() + rect.get_width() / 2, height + 5, f"{perc:.2f}%", ha="center", va="bottom", color='darkslategray')

#     plt.tight_layout()
#     plt.show()
#     return plt
# fig = create_plot_3(data_dict,df5)

# # Display the plot
# st.pyplot(fig)


import plotly.express as px

def create_interactive_radial_chart(data_dict, df5):
    key_main = "art_discipline"
    key_columns = data_dict[key_main]
    count_of_ones = df5[key_columns].sum()
    sorted_counts = count_of_ones.sort_values(ascending=False)
    sorted_counts.index = sorted_counts.index.str.replace('discip_', '')

    # Create a polar plot using Plotly Express
    fig = px.bar_polar(r=sorted_counts.values,
                       theta=sorted_counts.index,
                       color=sorted_counts.values,
                       color_continuous_scale='greens',  # Choose your preferred color scale
                       labels={'theta': 'Artistic Discipline', 'r': 'Count'},
                       title='Most Common Artistic Disciplines (Hover on the chart for Interactive Insights)',
                       )

    # Update layout for a better appearance
    fig.update_traces(text=sorted_counts.values,
                      hoverinfo='text',
                      marker=dict(line=dict(color='black', width=1)),
                      )
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=False,
                      margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins for size
                      height=600,  # Set height of the chart
                      width=800    # Set width of the chart
                      )

    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=False)
    return fig

# Usage:
fig = create_interactive_radial_chart(data_dict, df5)
st.plotly_chart(fig)

st.write("Visual Arts and Music are the most common disciplines, with Visual Arts being the predominant field, comprising 39.08% of the artists surveyed. This data is crucial for cultural institutions, policymakers, and arts organizations as it highlights the need for a balanced approach in supporting a diverse range of artistic disciplines.")
st.markdown("")
st.markdown("")
# st.markdown("fun fact: ")
# import matplotlib.pyplot as plt
# import numpy as np

# def create_radial_chart_counts(data_dict, df5):
#     key_main = "art_discipline"
#     key_columns = data_dict[key_main]
#     count_of_ones = df5[key_columns].sum()
#     sorted_counts = count_of_ones.sort_values(ascending=False)
#     sorted_counts.index = sorted_counts.index.str.replace('discip_', '')

#     # Create a radial chart based on counts
#     plt.figure(figsize=(8, 8))
#     ax = plt.subplot(111, polar=True)

#     # Set the theta values for each discipline
#     theta = np.linspace(0, 2 * np.pi, len(sorted_counts), endpoint=False)

#     # Convert counts to radii for the radial bars
#     radii = sorted_counts.values

#     # Create bars on the polar plot
#     bars = ax.bar(theta, radii, width=0.4, color='crimson', alpha=0.7)

#     # Adjusting the angle for labels
#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)

#     # Adding labels
#     plt.xticks(theta, sorted_counts.index, fontsize=8)

#     # Display counts on top of the bars
#     for bar, label in zip(bars, sorted_counts):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2, height + 5, f"{label}", ha='center', va='bottom')

#     plt.title("Most Common Artistic Disciplines (Radial Chart)", fontsize=14, fontweight='bold')
#     plt.yticks([])
#     plt.tight_layout()
#     plt.show()
#     return plt

# # Usage:
# fig = create_radial_chart_counts(data_dict, df5)
# st.pyplot(fig)

#######################################


##########################


st.subheader("Financial Stability")

st.write("The financial stability visualizations below offer a comprehensive view of artists income sources, forms of aid received, income levels, and debt manageability.")



# def create_plot_4(data_dict,df5):
#     # Assuming 'data_dict' is your dictionary and 'df5' is your DataFrame
#     earning_info_columns = data_dict['earning_info']  # Replace with actual dictionary access if different

#     # Calculate the sum (or count of 1s) for each column in earning_info
#     earning_info_sums = df5[earning_info_columns].sum()

#     # Sort the sums in descending order
#     sorted_sums = earning_info_sums.sort_values(ascending=False)
    
#     # Remove "earn_" prefix from the index labels
#     sorted_sums.index = sorted_sums.index.str.replace('earn_', '')

#     # Create a bar chart
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x=sorted_sums.values, y=sorted_sums.index, palette='viridis')

#     # Add titles and labels
#     plt.title('Artists Earning form')
#     plt.xlabel('Number of Artists →')
#     plt.ylabel('Type of Earnings')

#     plt.tight_layout()
#     plt.show()
#     return plt

# fig = create_plot_4(data_dict,df5)

# # Display the plot
# st.pyplot(fig)

import matplotlib.pyplot as plt
import seaborn as sns

def create_plot_4(data_dict, df5):
    # Assuming 'data_dict' is your dictionary and 'df5' is your DataFrame
    earning_info_columns = data_dict['earning_info']  # Replace with actual dictionary access if different

    # Calculate the sum (or count of 1s) for each column in earning_info
    earning_info_sums = df5[earning_info_columns].sum()

    # Sort the sums in descending order
    sorted_sums = earning_info_sums.sort_values(ascending=False)
    
    # Remove "earn_" prefix from the index labels
    sorted_sums.index = sorted_sums.index.str.replace('earn_', '')

    # Create a bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_sums.values, y=sorted_sums.index, palette='viridis')

    # Add titles and labels
    plt.title('Artists Earning from Different Sources')
    plt.xlabel('Number of Artists →')
    plt.ylabel('Type of Earnings')

    # Add annotations to each bar
    for i, value in enumerate(sorted_sums):
        plt.text(value, i, f' {value} ', ha='left', va='center', color='black', fontweight='bold')

    plt.tight_layout()
    return plt

fig = create_plot_4(data_dict, df5)

# Display the plot
st.pyplot(fig)

st.write("The bar chart provides an overview of the earning methods for artists. The most common earning method is through gigs and contracts, followed by part-time jobs, indicating a gig economy trend among artists. Unemployment and reliance on family support also appear significant, reflecting the economic vulnerability in the artistic profession. Full-time jobs are less common, suggesting that artists may prefer or require flexibility that traditional employment may not offer.")
st.markdown("")
st.markdown("")
st.markdown("")

# def create_plot_5(data_dict,df5):
#     # Set up the visualisation settings
#     sns.set(style="whitegrid")

#     assistance_columns = [col for col in df5.columns if 'assistance_' in col]
#     assistance_columns=assistance_columns[:len(assistance_columns)-1]
#     # assistance_columns
#     assistance_data = df5[assistance_columns].sum().sort_values()

#     # Create a bar chart
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x=assistance_data.values, y=assistance_data.index, palette='magma')

#     # Add titles and labels
#     plt.title('Types of Assistance Received by Artists')
#     plt.xlabel('Number of Artists')
#     plt.ylabel('Types of Assistance')

#     plt.tight_layout()
#     plt.show()
#     return plt

# fig = create_plot_5(data_dict,df5)

# # Display the plot
# st.pyplot(fig)



# st.write("ADD_CONTENT_HERE")

import pandas as pd
import plotly.express as px
import streamlit as st

def create_plot_6(data_dict, df5):
    # Your data preparation (similar to the code you provided)
    assistance_columns = [col for col in df5.columns if 'assistance_' in col]
    assistance_columns = assistance_columns[:len(assistance_columns) - 1]
    assistance_data = df5[assistance_columns].sum().sort_values()

    # Create a dataframe from the sorted data
    data = pd.DataFrame({'Assistance Type': assistance_data.index, 'Count': assistance_data.values})
    
    data['Assistance Type'] = data['Assistance Type'].str.replace('assistance_', '')

    # Create the interactive packed bubble chart using Plotly Express
    fig = px.treemap(data, path=['Assistance Type'], values='Count')

    # Update layout for better visualization (optional)
    fig.update_layout(title='Types of Assistance Received by Artists (Interactive)',height=500, width=800)

    return fig

fig = create_plot_6(data_dict, df5)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

st.write("The chart above illustrates various forms of aid artists have received. Unemployment benefits are by far the most common, followed by federal relief, indicating that government programs are a crucial support for artists, particularly in times of crisis. Other forms of assistance, such as emergency grants, personal or family assistance, and mutual aid, are less common but still significant. This suggests that while public assistance is vital, there is also a role for community and familial support networks in sustaining artists.")

st.markdown("")
st.markdown("")
st.markdown("")
import matplotlib.pyplot as plt
import seaborn as sns

def create_income_area_plot(data_dict, df5):
    # Define the order of income categories and display names
    income_order = [
        'Under $15,000', '$15,000 to $24,999', '$25,000 to $34,999',
        '$35,000 to $49,999', '$50,000 to $74,999', 
        '$75,000 to $99,999', '$100,000 to $149,999', 'Over $150,000',
    ]
    
    display_names = [
        'Under $15K', '$15K - $25K', '$25K - $40K',
        '$35K - $50K', '$50K - $75K', 
        '$75K - $100K', '$100K - $140K', 'Over $150K',
    ]
    
    # Create a DataFrame with counts for each income category
    income_counts = df5['p9_2021hhincome'].value_counts().loc[income_order].reset_index()
    income_counts.columns = ['Income Range', 'Count']
    plt.figure(figsize=(12, 6))
    sns.set_palette("viridis")  # Set color palette if desired

    plt.fill_between(income_counts['Income Range'], income_counts['Count'], color='skyblue', alpha=0.5)
    plt.plot(income_counts['Income Range'], income_counts['Count'], marker='o', color='blue', linewidth=2)

    # Adding labels to points
    for i, count in enumerate(income_counts['Count']):
        plt.text(i, count, str(count), ha='left', va='bottom', fontsize=12)

    # Adding labels and title
    plt.title('Distribution of Household Income for 2021')
    plt.xlabel('2021 Household Income')
    plt.ylabel('Count of Artists →')
    plt.xticks(range(len(income_order)), display_names, rotation=0, ha='center', size=10)
    plt.grid(False)  # Remove gridlines

    plt.tight_layout()
    return plt

fig = create_income_area_plot(data_dict, df5)

# Display the Plotly figure in Streamlit
st.pyplot(fig)
st.write("The bar chart portrays the income levels among a group of artists. The majority have an annual household income below USD 40K with the largest number falling under the USD 15K bracket. As household income increases, the number of artists within each income bracket decreases, indicating that higher earnings are less common.")
st.markdown("")
st.markdown("")
st.markdown("")
# Example usage:
# create_income_area_plot(income_order, display_names, income_counts)

# def finance_one(df5,data_dict):
    
#     # Count the frequency of each category in 'p14b_debtmanageable'
#     debt_manageability_counts = df5['p14b_debtmanageable'].value_counts()
    
#     # Calculate the total number of responses excluding blanks
#     total_responses = debt_manageability_counts.sum()
    
#     # Create a figure and axis without any background
#     fig, ax = plt.subplots(figsize=(8, 2), frameon=False)
#     ax.axis('off')
    
#     # Define the two categories and their counts
#     categories = ['Manageable', 'Unmanageable']
#     counts = [debt_manageability_counts.get(cat, 0) for cat in categories]
    
#     # Create two segments in a single horizontal bar with labels
#     bar = ax.barh(categories, counts, color=['blue', 'red'])
#     for rect, label in zip(bar, counts):
#         width = rect.get_width()
#         ax.text(width, rect.get_y() + rect.get_height() / 2, label, ha='left', va='center', fontsize=12, fontweight='bold')
    
#     # Add the total number of responses as text
#     ax.text(total_responses / 2, -0.5, f"Total Responses: {total_responses}", ha='center', va='center', fontsize=12, fontweight='bold')
    
#     # Show the chart
#     plt.title('Debt Manageability')
#     return plt
# fig = finance_one(df5,data_dict)

# # Display the Plotly figure in Streamlit
# st.pyplot(fig)


def finance_one(df5, data_dict):
    # Count the frequency of each category in 'p14b_debtmanageable'
    debt_manageability_counts = df5['p14b_debtmanageable'].value_counts()
    
    # Calculate the total number of responses excluding blanks
    total_responses = debt_manageability_counts.sum()
    
    # Create a figure and axis without any background
    fig, ax = plt.subplots(figsize=(8, 2), frameon=False)
    ax.axis('off')
    
    # Define the two categories and their counts
    categories = ['Manageable', 'Unmanageable']
    counts = [debt_manageability_counts.get(cat, 0) for cat in categories]
    
    # Create two segments in a single horizontal bar with labels
    bar = ax.barh(categories, counts, color=['blue', 'red'])
    for rect, label in zip(bar, counts):
        width = rect.get_width()
        ax.text(width, rect.get_y() + rect.get_height() / 2, f"{label}", ha='left', va='center', fontsize=12, fontweight='bold')
    
    # Add the total number of responses as text
    ax.text(total_responses / 2, -0.5, f"Total Responses: {total_responses}", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Set y-axis tick labels as categories
    ax.set_yticklabels(categories, fontsize=12)
    
    # Show the legend for colors with labels
    ax.legend(categories, loc='lower right', fontsize=10)
    
    # Show the chart
    plt.title('Debt Manageability')
    return plt

fig = finance_one(df5, data_dict)

# Display the plot
st.pyplot(fig)
st.markdown("")
st.markdown("")
st.markdown("")


st.write("The chart presents data on debt manageability, clearly showing two categories: those who find their debt unmanageable and those who find it manageable. With a total of 9544 responses, a significant majority, about 63%, indicate they find their debt unmanageable. This suggests a pressing financial concern and the potential need for financial advisory services, debt management assistance, or other supportive measures to help individuals manage their debt more effectively.")
st.markdown("")
st.markdown("")
st.markdown("")
st.subheader("Wellbeing of artists")
st.write("The well-being visualizations below shed light on the multifaceted factors impacting artists mental and emotional health, highlighting prevalent issues such as anxiety, loneliness, and economic strains.")

def create_plot_7(data_dict,df5):
    # Set up the visualisation settings
    sns.set(style="whitegrid")


    # Calculate the sum of each 'Wellbeing_Impact_' column to see the impact on artists' wellbeing
    wellbeing_columns = [col for col in df5.columns if 'Wellbeing_Impact_' in col]
    wellbeing_data = df5[wellbeing_columns].sum().sort_values()
    
    wellbeing_data.index = wellbeing_data.index.str.replace('Wellbeing_Impact_', '')

    # Create a bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x=wellbeing_data.values, y=wellbeing_data.index, palette='magma')

    # Add titles and labels
    plt.title('Wellbeing_Impact of Artists ')
    plt.xlabel('Number of Artists →')
    plt.ylabel('Impacted by')

    plt.tight_layout()
    plt.show()
    return plt

fig = create_plot_7(data_dict,df5)

# Display the plot
st.pyplot(fig)
st.markdown("")
st.markdown("")
st.markdown("")



# st.write("ADD_CONTENT_HERE")
# def create_plot_8(data_dict,df5):
#     # Set up the visualisation settings
#     sns.set(style="whitegrid")


#     # Calculate the sum of each 'Wellbeing_Impact_' column to see the impact on artists' wellbeing
#     wellbeing_columns = [col for col in df5.columns if 'Wellbeing_Impact_' in col]
#     wellbeing_data = df5[wellbeing_columns].sum().sort_values()
    
#     # Calculate percentage distribution
#     total_count = sum(wellbeing_data)
#     percentage_distribution = wellbeing_data / total_count * 100
#     percentage_distribution.index = percentage_distribution.index.str.replace('Wellbeing_Impact_', '')


#     # Create a bar chart
#     plt.figure(figsize=(12, 8))
#     # plt.figure(figsize=(8, 8))
#     plt.pie(percentage_distribution, labels=percentage_distribution.index, autopct='%1.1f%%',  startangle=0)
#     plt.title('Percentage Distribution of Wellbeing_Impact among Artists')
#     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     # plt.show()
#     return plt

# fig = create_plot_8(data_dict,df5)
# # Display the plot
# st.pyplot(fig)

st.write("This bar chart provides insights into various factors affecting artists' well-being. Notably, 'Anxiety' and 'Loneliness' are the most reported impacts, highlighting significant mental health challenges within the artist community. Issues like 'Accumulated Debt' and 'Housing Insecurity' , indicating economic strains. This data is critical for organizations aiming to support artists, indicating a need for mental health resources, financial counseling, and social connectivity initiatives.")
st.markdown("")
st.markdown("")
st.markdown("")
import seaborn as sns
import matplotlib.pyplot as plt

def create_plot_9(data_dict,df5):

    # Define the mapping of old column names to new names
    name_mapping = {
        "p15_physicalhealth": "Physical Health",
        "p16_mentalhealth": "Mental & Emotional Health",
        "p17_stablehousing": "Housing Needs",
        "p18_feedmyself": "Food Availibility",
        "p19_socialrelationships": "Social Relationships",
        "p20_purposefullife": "Purposeful & Meaningful Life",
        "p21_agency": "Agency Over Future",
        "p22_optimistic": "Optimistic Future",
    }
    
    # Rename the columns in the DataFrame
    df5_renamed = df5.rename(columns=name_mapping)
    
    # Compute the average rating for each attribute in the renamed DataFrame
    avg_ratings = df5_renamed[list(name_mapping.values())].mean()
    
    # Sort the average ratings in ascending order
    sorted_avg_ratings = avg_ratings.sort_values(ascending=False)
    
    # Create a DataFrame for the heatmap
    heatmap_data = pd.DataFrame(sorted_avg_ratings).T
    
    
    
    # Create a horizontal heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(heatmap_data.T, annot=True, cmap="RdYlGn", cbar_kws={'label': 'Average Rating'})
    plt.title("Average overall well-being of artists")
    plt.ylabel("Well being attributes")
    plt.xlabel("Rating Scale")
    plt.xticks(rotation=0)  # No need to rotate x-axis labels
    plt.yticks(rotation=0)  # Make y-axis labels horizontal for readability
    return plt

fig = create_plot_9(data_dict,df5)
# Display the plot
st.pyplot(fig)
st.write("The above visualization is about understanding the well-being of artists in New York State. The data suggests that while artists find a high level of satisfaction in leading a purposeful and meaningful life and maintaining social relationships, there are challenges in areas like Mental & Emotional Health and Housing Needs, which have received the lowest average ratings.")

st.markdown("")
st.markdown("")

st.subheader("Pandemic Impact")
st.markdown("***How did the pandemic impact employment?***")
st.write("The pandemic had significant impacts on employment within the artistic community, with notable disruptions such as the cessation of collaborations and the cancellation of shows.")


# import matplotlib.pyplot as plt
# import numpy as np

# def convert_to_radial_bar_chart(data_dict, df):
#     key_main = "practice_impact"
#     key_columns = data_dict[key_main]

#     # Calculate the count of 1s in each column
#     count_of_ones = df[key_columns].sum()

#     # Sort the columns by descending order of the count of 1s
#     sorted_columns = count_of_ones.sort_values(ascending=False).index

#     # Calculate the total count for percentage calculation
#     total_count = count_of_ones.sum()

#     # Calculate the percentage for each category
#     percentage_values = [(count / total_count) * 100 for count in count_of_ones[sorted_columns]]

#     # Remove "impact_" from the labels
#     cleaned_labels = [label.replace("practice_impact_", "") for label in sorted_columns]

#     # Define a custom color palette
#     colors = plt.cm.viridis(np.linspace(0, 1, len(cleaned_labels)))  # Change the colormap as needed

#     # Number of categories
#     num_categories = len(cleaned_labels)

#     # Set the figure size
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

#     # Compute the width of each bar
#     width = 2 * np.pi / num_categories

#     # Plot each bar as a segment in the polar plot
#     bars = ax.bar(np.linspace(0, 2 * np.pi, num_categories, endpoint=False), percentage_values, width=width, bottom=0.0, color=colors)

#     # Display values on the bars
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2, height, f"{percentage_values[i]:.2f}%", ha='center', va='bottom', fontsize=8)

#     # Set the labels for each bar
#     ax.set_xticks(np.linspace(0, 2 * np.pi, num_categories, endpoint=False))
#     ax.set_xticklabels(cleaned_labels, fontsize=10)

#     # Set title for the radial bar chart
#     plt.title('Pandemic Impact on Artists Employment', fontsize=14)

#     return fig

# # Example usage in Streamlit
# fig = convert_to_radial_bar_chart(data_dict, df5)
# st.pyplot(fig)


# st.write("ADD_CONTENT_HERE")


# def convert_to_radial_bar_chart(data_dict, df):
#     key_main = "practice_impact"
#     key_columns = data_dict[key_main]

#     # Calculate the count of 1s in each column
#     count_of_ones = df[key_columns].sum()

#     # Sort the columns by descending order of the count of 1s
#     sorted_columns = count_of_ones.sort_values(ascending=False).index

#     # Remove "impact_" from the labels
#     cleaned_labels = [label.replace("practice_impact_", "") for label in sorted_columns]

#     # Create a DataFrame for plotting
#     data = pd.DataFrame({
#         'Labels': cleaned_labels,
#         'Values': count_of_ones[sorted_columns]
#     })

#     # Create a Plotly bar polar chart
#     fig = go.Figure()

#     fig.add_trace(go.Barpolar(
#         r=data['Values'],
#         theta=data['Labels'],
#         text=data['Values'],
#         hoverinfo='theta+text',
#         marker=dict(color='green'),  # Change the color as needed
#     ))

#     fig.update_layout(
#         title='Pandemic Impact on Artists Employment',
#         polar=dict(radialaxis=dict(visible=True)),
#         showlegend=False,
#         polar_bgcolor='rgba(0,10,9,0)',  # Remove the grid
#         polar_radialaxis=dict(showline=False),  # Remove the radial axis line
#         height=600,  # Increase the height
#         width=800,  # Increase the width
#     )

#     return fig

# # Example usage in Streamlit
# fig = convert_to_radial_bar_chart(data_dict, df5)
# st.plotly_chart(fig)

import seaborn as sns
import matplotlib.pyplot as plt
def imapct_method(data_dict, df5):
    
    key_main = "practice_impact"
    key_columns = data_dict[key_main]
    
    # Calculate the count of 1s in each column
    count_of_ones = df5[key_columns].sum()
    
    # Sort the columns by descending order of the count of 1s
    sorted_columns = count_of_ones.sort_values(ascending=False).index
    
    # Calculate the total count for percentage calculation
    total_count = count_of_ones.sum()
    
    # Calculate the percentage for each category
    percentage_values = [(count / total_count) * 100 for count in count_of_ones[sorted_columns]]
    
    # Remove "impact_" from the x-axis parameter labels
    cleaned_labels = [label.replace("practice_impact_", "") for label in sorted_columns]
    
    # Create a horizontal bar plot with fading colors
    plt.figure(figsize=(15, 9))
    palette = sns.color_palette("Blues_r", len(cleaned_labels))  # Fading colors from high to low
    ax = sns.barplot(x=count_of_ones[sorted_columns], y=cleaned_labels, orient="h", palette=palette)
    
    # Annotate the count and percentage of each category inside the bar
    for i, (count,) in enumerate(zip(count_of_ones[sorted_columns])):
        text = f"{count}"
        ax.text(count + 5, i, text, va='center', fontsize=12)  # Adjust the position for better readability
    
    plt.title('Pandemic impact on Artists employment')  # Updated title
    plt.xlabel('Count of Artists →')
    plt.ylabel('Impacts  →')
    plt.xticks( ha="right")
    # plt.show()
    return plt

fig = imapct_method(data_dict, df5)
st.pyplot(fig)



st.write("The data reveals that the most substantial impacts include the cessation of collaborations and the cancellation of shows, indicating major disruptions to artists usual means of income and engagement with the art community.")
st.markdown("")
st.markdown("")
st.markdown("")
st.subheader("Geographic and Demographic Information")

st.write("The geographic and demographic information provides insights into awareness levels of guaranteed income policies across different communities, concentration of artists in various cities, gender distribution, racial diversity among artists, and the causes artists are associated with, guiding targeted support and inclusive initiatives.")

def geo_one(df,data_dict):
    # Replacing "," s with Null values
    df5['community'].unique()
    x = [',']
    y = [np.NaN]
    df5.replace(x, y, inplace=True)
    df5['community'].unique()
    
    # Define the desired order of communities
    community_order = ["Tribal", "Suburban", "Urban", "Rural"]
    
    # Apply the custom order to the 'community' column
    df5['community'] = pd.Categorical(df5['community'], categories=community_order, ordered=True)
    
    # Define the language and awareness columns
    language_column = 'community'
    awareness_column = 'p26_awareofgi'
    
    # Create a new DataFrame with relevant columns
    df_language_awareness = df5[[language_column, awareness_column]]
    
    # Group the data by primary spoken language and awareness responses
    language_awareness_counts = df_language_awareness.groupby([language_column, awareness_column]).size().unstack(fill_value=0)
    
    # Normalize the counts to get percentages within each language group
    language_awareness_percentage = language_awareness_counts.divide(language_awareness_counts.sum(axis=1), axis=0) * 100
    
    # Reorder the columns to make the plot more interpretable
    awareness_order = ["Yes", "No", "Not sure"]
    language_awareness_percentage = language_awareness_percentage[awareness_order]
    
    # Create the stacked bar chart with the custom order
    # plt.figure(figsize=(12, 6))
    # language_awareness_percentage.plot(kind='bar', stacked=True, colormap='Set2')
    
    # plt.xlabel('Community Type')
    # plt.ylabel('Percentage (%)')
    # plt.title('Awareness of Guaranteed Income Policies by Community Type')
    
    # plt.legend(title='Awareness', loc='upper left', labels=['Yes', 'No', 'Not sure'])
    
    # plt.tight_layout()
    plt.figure(figsize=(15, 9))
    language_awareness_percentage.plot(kind='bar', stacked=True, colormap='Set2')
    
    plt.xlabel('Community Type')
    plt.ylabel('Percentage (%)')
    plt.title('Awareness of Guaranteed Income Policies by Community Type')
    
    # Place the legend outside the plot
    plt.legend(title='Awareness', loc='upper left', labels=['Yes', 'No', 'Not sure'], bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.tight_layout()
    # plt.show()
    return plt
fig = geo_one(data_dict, df5)
st.pyplot(fig)



st.write("The above chart reveals that Urban communities have the highest awareness levels, while Tribal communities have the lowest. Suburban and Rural communities show similar levels of awareness. This data suggest a need for targeted information campaigns in Tribal communities to raise awareness about guaranteed income policies.")
st.markdown("")
st.markdown("")
st.markdown("")

def geo_two(df5,data_dict):
    # Select the top N cities
    top_n = 10
    top_cities = df5['city'].value_counts().head(top_n)
    
    # Create a new DataFrame for the top cities
    df_top_cities = top_cities.reset_index()
    df_top_cities.columns = ['City', 'Count']
    
    # Determine bubble sizes
    bubble_sizes = df_top_cities['Count'] * 10  # Adjust this factor as needed
    
    # Create the bubble chart for top cities
    plt.figure(figsize=(20, 15))
    plt.scatter(df_top_cities['City'], df_top_cities['Count'], s=bubble_sizes, alpha=1, color='skyblue')
    plt.title(f'Distribution of Respondents by Top {top_n} Cities')
    plt.xlabel('City')
    plt.ylabel('Count of Artists →')
    plt.xticks(rotation=00, ha='right')
    # plt.show()
    return plt

fig = geo_two(df5,data_dict)
st.pyplot(fig)

st.write("The bubble chart illustrates the concentration of Artists various cities.  Brooklyn having the largest representation. New York (likely referring to Manhattan) indicating a higher concentration of artists in these urban areas. This visual suggests that artistic communities are more densely populated in certain urban areas, which could be vital for targeted cultural investment and support initiatives.")
st.markdown("")
st.markdown("")
st.markdown("")
def gender_chart(df5, data_dict):
    import matplotlib.pyplot as plt

    # Assuming df5 and gender_columns are defined as before
    gender_columns = ['gender_Man', 'gender_Non_binary', 'gender_Prefer_not_to_answer_gender', 'gender_Prefer_other', 'gender_Two_spirit', 'gender_Woman']

    gender_counts = df5[gender_columns].sum()

    # Remove 'gender_' from the column labels for the chart
    gender_labels = [label.replace('gender_', '') for label in gender_columns]

    # Create a doughnut chart with percentages
    fig, ax = plt.subplots()
    ax.pie(gender_counts, labels=gender_labels, startangle=90, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'], wedgeprops=dict(width=0.4), autopct='%1.1f%%')

    # Draw a circle at the center of pie to make it look like a doughnut
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie/doughnut is drawn as a circle.
    ax.axis('equal')
    plt.title("Gender Distribution - Doughnut Chart")

    plt.show()
    return plt

fig = gender_chart(df5, data_dict)
st.pyplot(fig)
st.write("The doughnut chart Gender Distribution indicates that a majority of the subjects identify as Man (56.9%), with Woman being the next largest group (29.7%). Smaller segments of the population identify as Non_binary, Two_spirit,Prefer_other, or prefer not to specify their gender, highlighting the presence and recognition of diverse gender identities within this group.")
st.markdown("")
st.markdown("")
st.markdown("")

def ethinicity_(df5,data_dict):
    key_main = "ethinicity"  # Updated key to "ethinicity"
    key_columns = data_dict[key_main]
    
    # Calculate the count of 1s in each column
    count_of_ones = df5[key_columns].sum()
    
    # Remove "Race_Ethnicity_" from the attribute names
    sorted_columns = count_of_ones.index.str.replace("ethnicity_", "")
    
    # Create a DataFrame for the plot
    df_plot = pd.DataFrame({'Columns': sorted_columns, 'Count of 1s': count_of_ones.values})
    
    # Calculate the percentage
    df_plot['Percentage'] = (df_plot['Count of 1s'] / df_plot['Count of 1s'].sum()) * 100
    
    # Create a donut chart using Plotly
    fig = px.pie(df_plot, names='Columns', values='Percentage', hole=0.4, title=f'Artists distribution as per {key_main} (Interactive)')
    
    # Show the chart
    return fig
fig =ethinicity_(df5,data_dict)
# Display the Plotly figure in Streamlit
st.plotly_chart(fig)


st.write("The above doughnut chart provides a visual representation of the racial diversity among a group of artists Predominant Groups: The largest portion of the artists identifies as White (31.7%), followed by Black or African American (24.9%). Minority Representation: Smaller percentages of artists identify as Asian (8.47%), Indigenous American/First Nation (4.08%). This visualization is a valuable tool for understanding the racial composition of artists in the sample and underscores the importance of fostering a diverse and inclusive arts community.")
st.markdown("")
st.markdown("")
st.markdown("")

def social_status(df5,data_dict):

    # Define the names of the columns
    causes_columns = [
        'p27_causes_Arts_Culture',
        'p27_causes_Childcare_Access',
        'p27_causes_Disability_Justice',
        'p27_causes_Economic_Justice',
        'p27_causes_Environment_Climate_Justice',
        'p27_causes_Housing_Tenants_Rights',
        'p27_causes_Labor_Workers_Rights',
        'p27_causes_None_At_This_Time',
        'p27_causes_Other_Causes',
        'p27_causes_Social_Racial_Justice'
    ]
    
    # Replace the prefix with an empty string to clean up the labels
    causes_labels = [cause.replace('p27_causes_', '') for cause in causes_columns]
    
    # Create a DataFrame with the selected columns
    df_causes = df5[causes_columns]
    
    # Calculate the sum of each column to get the count
    causes_counts = df_causes.sum()
    
    # Create a DataFrame for the plot
    df_plot = pd.DataFrame({'Causes': causes_labels, 'Count': causes_counts})
    
    # Create a donut chart using Plotly Express
    fig = px.pie(df_plot, names='Causes', values='Count', hole=0.2, title='Participation in Coalitions and Causes (Interactive)')
    return fig

fig = social_status(df5,data_dict)
st.plotly_chart(fig)

st.write("The doughnut chart shows the various causes that Artists are associated with. The chart indicates a diverse range of interests among the respondents, but with a strong leaning towards cultural and social justice issues. This distribution can guide organizations in aligning their programs with the causes that resonate most with their audience.")
# import plotly.graph_objects as go
st.markdown("")
st.markdown("")
st.markdown("")

# def social_status_radar(df5, data_dict):
#     causes_columns = [
#         'p27_causes_Arts_Culture',
#         'p27_causes_Childcare_Access',
#         'p27_causes_Disability_Justice',
#         'p27_causes_Economic_Justice',
#         'p27_causes_Environment_Climate_Justice',
#         'p27_causes_Housing_Tenants_Rights',
#         'p27_causes_Labor_Workers_Rights',
#         'p27_causes_None_At_This_Time',
#         'p27_causes_Other_Causes',
#         'p27_causes_Social_Racial_Justice'
#     ]
    
#     causes_labels = [cause.replace('p27_causes_', '') for cause in causes_columns]
    
#     df_causes = df5[causes_columns]
#     causes_counts = df_causes.sum()
#     df_plot = pd.DataFrame({'Causes': causes_labels, 'Count': causes_counts})
    
#     fig = go.Figure()

#     fig.add_trace(go.Scatterpolar(
#         r=df_plot['Count'],
#         theta=df_plot['Causes'],
#         fill='toself',
#         name='Participation in Coalitions and Causes',
#         line=dict(color='green')  # Change the line color as needed
#     ))

#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, max(df_plot['Count']) + 10]
#             )
#         ),
#         showlegend=True,
#         title='Participation in Coalitions and Causes'
#     )

#     return fig

# fig = social_status_radar(df5, data_dict)
# st.plotly_chart(fig)


# import plotly.graph_objects as go

# def social_status_radar(df5, data_dict):
#     causes_columns = [
#         'p27_causes_Arts_Culture',
#         'p27_causes_Childcare_Access',
#         'p27_causes_Disability_Justice',
#         'p27_causes_Economic_Justice',
#         'p27_causes_Environment_Climate_Justice',
#         'p27_causes_Housing_Tenants_Rights',
#         'p27_causes_Labor_Workers_Rights',
#         'p27_causes_None_At_This_Time',
#         'p27_causes_Other_Causes',
#         'p27_causes_Social_Racial_Justice'
#     ]
    
#     causes_labels = [cause.replace('p27_causes_', '') for cause in causes_columns]
    
#     df_causes = df5[causes_columns]
#     causes_counts = df_causes.sum()
#     df_plot = pd.DataFrame({'Causes': causes_labels, 'Count': causes_counts})
    
#     fig = go.Figure()

#     fig.add_trace(go.Scatterpolar(
#         r=df_plot['Count'],
#         theta=df_plot['Causes'],
#         fill='toself',
#         name='Participation in Coalitions and Causes',
#         line=dict(color='purple')  # Change the line color as needed
#     ))

#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, max(df_plot['Count']) + 10]
#             )
#         ),
#         showlegend=True,
#         title='Participation in Coalitions and Causes',
#         polar_bgcolor='rgba(255, 255, 255, 0)',  # Set transparent background
#         polar_radialaxis_ticks='',
#         polar_angularaxis_ticks=''
#     )

#     # Remove gridlines
#     # fig.update_polars(radialaxis_showgrid=False, angularaxis_showgrid=False)

#     return fig

# fig = social_status_radar(df5, data_dict)
# st.plotly_chart(fig)

st.subheader("SUMMARY")

st.write("In the world of artistry, diversity and depth unfold as artists from various backgrounds and disciplines paint their unique stories on the canvas of life. Our journey through these visualizations reveals the tapestry of this artistic ecosystem. First, we delve into the essence of Artistic Approach, where solo endeavors and collaborative creations intertwine. The age distribution speaks of mid-career artists in their prime, while artistic disciplines like Visual Arts and Music form the vivid palette of expression. Financial Stability unveils the earning methods and support systems that sustain these talents. Gigs and contracts are the brushstrokes of income, and government programs provide essential strokes of support. Amidst it all, Debt Manageability casts a shadow. A majority grapples with unmanageable debt, beckoning for financial guidance and relief. Wellbeing, a delicate subject, reveals anxiety and loneliness as formidable adversaries, while purpose and social connections stand strong. Housing Insecurity and Mental Health yearn for brighter days. Pandemic Impact paints a tale of halted collaborations and canceled shows, leaving artists to navigate the storm's aftermath. Lastly, Geographic and Demographic Information sheds light on awareness levels, cityscapes adorned with artistry, gender and racial diversity, and the multitude of causes embraced. This artistic world, diverse and vibrant, beckons for support, understanding, and unity to preserve and celebrate the myriad stories it has yet to tell.")

st.subheader("Description of the methods used:")

"""***Data Exploration and Preprocessing:*** We began by thoroughly exploring the dataset, identifying columns with minimal impact or a majority of null values. These columns were removed to streamline the dataset.

***Data Regularization:*** To work with categorical data effectively, we employed various data preprocessing techniques, including one-hot encoding and label encoding, to convert categorical values into numerical representations.

***Data Dictionary:*** A data dictionary was established to map columns that contained survey responses to specific questions. This facilitated grouping and analysis of related data.

***Data Analysis:*** With the cleaned and processed data, we conducted in-depth data analysis to extract meaningful information. This involved calculating counts, percentages, and aggregations to gain insights into various aspects, such as artistic approaches, financial stability, well-being, and more.

***Visualization:*** The findings were transformed into visualizations to make the information more accessible and comprehensible. We used various visualization techniques, including bar charts, histograms, doughnut charts, and bubble charts, to represent different aspects of the data.

***Narrative Construction:*** Finally, we wove the insights from these visualizations into a cohesive narrative, creating a compelling story that conveyed the richness and diversity of the artistic community's experiences."""
