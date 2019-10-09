#RENAME
import os
path = 'D:/1042802/EVA/Assignment19/Faces'
files = os.listdir(path)
i = 1

for file in files:
    if (i<10):
        os.rename(file,"img_00" + str(i) +'.jpg')
    else:
        os.rename(file,"img_0" + str(i) +'.jpg')
    i = i+1    

#RESIZE    
from PIL import Image    
files = os.listdir(path)
#files.remove('yolo.py')

for file in files:
    img = Image.open(file)
    img = img.resize((400,400))
    img.save(file)


#RETRIVING HEIGHT AND WIDTH VALUES FROM JSON    
import pandas as pd
data = pd.read_json("via_project_26Sep2019_11h52m.json")
img_data = data['_via_img_metadata']
width_list = []
height_list = []

for key,value in img_data.items():
    if('img' in key):
        for i,j in value.items():
            if(isinstance(value[i],list)):
                for i in value[i]:
                    for a,b in i.items():
                        for c,d in b.items():
                            if('width' in c):
                                width_list.append(d)
                            elif('height' in c):
                                height_list.append(d)

#CONVERTING LIST INTO DATAFRAME
hw = pd.DataFrame()

norm_val = 400
new_height_list = [x/norm_val for x in height_list]

new_width_list = [x/norm_val for x in width_list]
hw['height'] = new_height_list
hw['width'] = new_width_list

hw

#APPLYING K-MEANS
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(hw)

#FINDING CLUSTER CENTERS
kmeans.cluster_centers_

