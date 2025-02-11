from flask import Flask, request, render_template

app = Flask(__name__)

# routes
@app.route('/')
def index(): 
    pass
    # return render_template('index.html')

# python main
if __name__ == "__main__":
    app.run(debug=True)