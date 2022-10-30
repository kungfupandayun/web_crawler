# Importing the required modules   

#pandas package needs to be downloaded
import pandas as pd 
import urllib.request, json
import matplotlib.pyplot as plt
from textwrap import wrap
with urllib.request.urlopen("https://www.enforcementtracker.com/data.json?") as url:
  jsondata=json.loads(url.read())
   
# empty list 
data_list = [] 
list_header=['Code','Country','Authority','Date','Fine','Controller','Sector','Quoted Art.','Type','Summary','Link']


for one_data in jsondata['data']: 
    sub_data = [] 
    for i in range(1,12): 
        try:
            if i==2 :
                one_data[i]=one_data[i].rsplit("<br />",1)
                one_data[i]=one_data[i][1]
            if i==11:
                one_data[i]=one_data[i].split("'")
                one_data[i]=one_data[i][3]                
            sub_data.append(one_data[i]) 
        except: 
            continue
    data_list.append(sub_data) 
  
# Storing the data into Pandas 
# DataFrame  
dataFrame = pd.DataFrame(data = data_list, columns = list_header) 
   
# Converting Pandas DataFrame 
# into CSV file 
dataFrame.to_csv('Geeks.csv')

# Finding Top 10 Countries
grouped_data=pd.DataFrame(dataFrame['Country'].value_counts())
grouped_data=grouped_data.reset_index()
grouped_data.columns=['Country', 'Counts']
top10_countries=grouped_data.head(10)
print(top10_countries)

# Exporting the Top 10 Countries Plot
top10_countries.plot(figsize=(10,6), x='Country', y='Counts', kind='bar' , rot=0)
plt.xlabel('Countries', fontsize='large', fontweight='bold')
plt.ylabel('Number of fines related to Countries', fontsize='large', fontweight='bold')
plt.xticks(fontsize='medium')
plt.tight_layout();
plt.savefig("../report_build/graphs_statistics/top10_countries.jpeg")

# Finding Top 10 Quoted Articles
quoted_data=pd.DataFrame(dataFrame['Quoted Art.'].value_counts())
quoted_data=quoted_data.reset_index()
quoted_data.columns=['Quoted Art.', 'Counts']
top10_quotedart=quoted_data.head(10)
print(top10_quotedart)

# Exporting the Top 10 Quoted Articles Plot
top10_quotedart['Quoted Art.']= top10_quotedart['Quoted Art.'].str.wrap(10)
top10_quotedart.plot(figsize=(10,6), x='Quoted Art.', y='Counts', kind='bar', rot=0)
plt.xlabel('Quoted Articles', fontsize='large', fontweight='bold')
plt.ylabel('Number of fines related to Articles', fontsize='large', fontweight='bold')
plt.tight_layout();
plt.savefig("../report_build/graphs_statistics/top10_quotedart.jpeg")

# Retrieving Sectors Overall Statistics
sector_data=pd.DataFrame(dataFrame['Sector'].value_counts())
sector_data=sector_data.reset_index()
sector_data.columns=['Sector', 'Counts']
print(sector_data)

# Exporting Sectors Overall Statistics Plot
sector_data['Sector']= sector_data['Sector'].str.wrap(18)
sector_data.plot(figsize=(11,7), y='Counts', kind='pie', labels=None, autopct='%1.0f%%', pctdistance=1.1)
plt.legend(labels=sector_data['Sector'], fontsize='large', loc="upper right", bbox_to_anchor=(1.5, 1.2), ncol=1)
plt.ylabel('')
plt.xlabel('')
plt.title('Sectors Overall Statistics', fontsize='large', fontweight='bold')
plt.tight_layout()
plt.savefig("../report_build/graphs_statistics/sector_data.jpeg")

# Finding Top 10 Controller / Number of fines
controller_data=pd.DataFrame(dataFrame['Controller'].value_counts())
controller_data=controller_data.reset_index()
controller_data.columns=['Controller', 'Counts']
top10_controller=controller_data.head(10)
print(top10_controller)

# Exporting the Top 10 Controller
top10_controller['Controller']= top10_controller['Controller'].str.wrap(10)
top10_controller.plot(figsize=(10,6), x='Controller', y='Counts', kind='bar', rot=0)
plt.xlabel('Controller', fontsize='medium', fontweight='bold')
plt.ylabel('Number of fines related to Controller', fontsize='large', fontweight='bold')
plt.tight_layout();
plt.savefig("../report_build/graphs_statistics/top10_controller.jpeg")

# Finding Top 10 Controller / Sum of fines
controller_per_fine=pd.DataFrame(columns=['Controller', 'Sum of fines'])
controller_sum_fine=[]
for controller in controller_data['Controller']:
    ex=dataFrame.loc[dataFrame['Controller']==controller,:]
    ex=ex.reset_index()
    ex=ex[["Fine"]]
    ex.columns=['Fine']

    ind=0
    for fine in ex['Fine']:
        if(fine=="Unknown"):
            ex = ex.drop(ind, axis=0)
        ind=ind+1
    sum=0
    for fine in ex['Fine']:
        fine=int(str(fine).replace(',',''))
        sum=sum+fine
    ex.reset_index()
    controller_sum_fine.append(sum)


sumoffines_per_controller={'Controller': controller_data["Controller"], 'Sum of Fines': controller_sum_fine}
sumoffines_per_controller=pd.DataFrame(sumoffines_per_controller)
sumoffines_per_controller.reset_index()
sumoffines_per_controller.columns=['Controller', 'Sum of Fines']

# Exporting the Top 10 Controller / Sum of fines
sumoffines_per_controller=sumoffines_per_controller.sort_values("Sum of Fines", ascending=False)
top10_sumoffines_per_controller=sumoffines_per_controller.head(10)
print(top10_sumoffines_per_controller)
top10_sumoffines_per_controller['Controller']= top10_sumoffines_per_controller['Controller'].str.wrap(10)
top10_sumoffines_per_controller.plot(figsize=(10,7), x='Controller', y='Sum of Fines', kind='bar', rot=0)
plt.xlabel('Controller', fontsize='large', fontweight='bold')
plt.ylabel('Sum of fines related to Controller', fontsize='large', fontweight='bold')
plt.tight_layout();
plt.savefig("../report_build/graphs_statistics/top10_controller_fines.jpeg")