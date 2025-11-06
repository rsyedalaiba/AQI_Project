*This folder contains python notebooks used in AQI Project.*

**How to Open .py Files as .ipynb**  

**Option 1: Open in Google Colab:**<br>

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

**Option 2: Open in Jupyter Notebook (Locally)**<br>

1. Download the .ipynb file to your computer (using colab).<br>
2. Open Anaconda Navigator or your terminal.<br>
3. Launch Jupyter Notebook.<br>
4. In your browser, navigate to where the file is saved.<br>
5. Click on the .ipynb file it will open and run like any normal Jupyter notebook.<br>

