# Import necessary libraries
from flask import Flask, request, render_template
from model import dataset,weightage,calculate_weighted_similarity

# Create Flask application
app = Flask(__name__)

# Define Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    
    try:
        # Retrieve input data from form
        search_for_recommendations = request.form['medname']
        
        # Obtain the Medicine from the dataset
        medicine = dataset[dataset['name'] == search_for_recommendations]

        # Obtain Compostition and Manufacturer
        composition = medicine['Disp'].values[0]
        manufacturer = medicine['manufacturer_name'].values[0]

        # Model Code
        if medicine['primary_comp'].values[0] == medicine['entire_comp'].values[0].rstrip():
            del weightage['primary_comp_embeddings']

        if medicine['value'].values[0] == medicine['entire_value'].values[0].rstrip():
            del weightage['value_embeddings']
        new_dataset = dataset.copy(deep=True)
        new_dataset['similarity_score'] = dataset.apply(lambda x: calculate_weighted_similarity(medicine, x), axis=1)
        top_10_similar_medicines = new_dataset.nlargest(11, 'similarity_score')[1:]

        # Rename Columns for WebPage
        top_10_similar_medicines.rename(columns={'name':'Name','Disp':'Composition','manufacturer_name':'Manufacturer Name','price(₹)': 'Price (₹)', 'similarity_score':'Similarity Score'}, inplace=True)

        ## Convert DF to HTML
        df_html = top_10_similar_medicines[['Name', 'Composition','Manufacturer Name','Price (₹)', 'Similarity Score']].to_html(border=1, classes='table table-striped', index=False)

        ## Render Template
        return render_template('index.html', medname=search_for_recommendations, composition = composition, manufacturer=manufacturer, data=df_html)

    except Exception as e:
        print("  \(>.<)/  --- ERROR NOT WORKING")
        print(f"An Error Occured: {e}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)