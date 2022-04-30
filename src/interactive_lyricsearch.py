
# !pip install ipywidgets

"""
Interactive Model
"""
import ipywidgets as widgets

query = widgets.Text(
        value='Sitting in my truck',
        description='Input query:')

button = widgets.Button(description='Submit')

slider = widgets.IntSlider(
         value=5,
         min=0,
         max=20,
         step=1,
         description='# of results:')

def on_click(_):
    with query:
        clear_output()
        print(query.value)
        

interact = widgets.TwoByTwoLayout(top_left=query,
                       bottom_left=slider)
interact

user_query = query.value
print("running query...")

query_embeddings = get_embeddings([user_query])[0]

# Return X nearest neighbors
nns = ann.get_nns_by_vector(query_embeddings, slider.value, include_distances=False)

print("Top {} results for \'{}\'\n".format(slider.value, query.value))
for idx, item in enumerate(nns):
    print("{}. {} - {}:".format(idx+1, mapping[item][1], mapping[item][2]))
    for x in range(item-3, item+3):
        if x == item:
            print("==== {} ====".format(mapping[x][0]))
        else:
            print("     {}     ".format(mapping[x][0]))
    
    print("\n")