This folder contains all .py scripts used in AQI Project.. 

**How to Open .py Files as .ipynb**  

**Option 1: Using VS Code or PyCharm:**<br>

1. Go to GitHub repo page.<br>
2. Click the green “Code” button → Download ZIP.<br>
3. Extract the ZIP file.<br>
4. Open the folder in VS Code or PyCharm.<br>
5. In the terminal, run:<br>
   python your_script_name.py<br>
This runs the .py file just like any Python program.<br>

**Option A: Open in Google Colab:**<br>

If you want to open and view the Python script as Notebook in Colab:<br>

1. ***Upload the .py file to Google Colab using:***<br>
        from google.colab import files<br>
        uploaded = files.upload()<br>
        
3. ***Install and use the converter:***<br>
     !pip install p2j<br>
     !mv "your_file_name.py" your_file_name.py<br>
     !p2j your_file_name.py<br>
4. ***Download and open the notebook:***<br>
     from google.colab import files<br>
     files.download("your_file_name.ipynb")<br>
5. ***Open the downloaded .ipynb file directly in Google Colab.***<br>

**Option B: Open in Jupyter Notebook (Locally)**<br>

1. Download the .ipynb file to your computer (using colab).<br>
2. Open Anaconda Navigator or your terminal.<br>
3. Launch Jupyter Notebook.<br>
4. In your browser, navigate to where the file is saved.<br>
5. Click on the .ipynb file it will open and run like any normal Jupyter notebook.<br>
