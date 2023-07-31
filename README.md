# Summarization-Project-using-Pointer-Gen

Paper link: https://arxiv.org/pdf/1704.04368v2.pdf

Dataset link: kaggle link

*Used the Pointer Generator Network for coming up with this summarization technique which turns out to give better ROUGE score than the simple encoder and decoder model. Using Pointer also improved the UNK tokens which appeared a lot and to cover up the problem of repetation in the summary I employed coverage mechanism.*

## Save the CNN and Dailymail stories in github (Not necessary)
**Step 1: Download gitpython**

`
!pip install gitpython
`

**Step 2: Import libraries**

```python
import os
from git import Repo
```

**Step 3: Clone repository**

```python
username = "Arup3201"
repository = "Summarization-Project-using-Pointer-Gen"
repository_url = f"https://github.com/{username}/{repository}.git"
repo_dir = "Summarization-Project-using-Pointer-Gen"  # Local directory to clone the repository
Repo.clone_from(repository_url, repo_dir)
```

**Step 4: Download the datasets using `tf.keras.utils.get_file`**

```python
import tensorflow as tf

# Download the CNN stories from the url into cnn_stories_tgz file
cnn_stories_tgz = tf.keras.utils.get_file(
    origin="https://huggingface.co/datasets/cnn_dailymail/resolve/main/data/cnn_stories.tgz",
    extract=True
)

# Download the Dailymail stories from the url into dailymail_stories_tgz file
dailymail_stories_tgz = tf.keras.utils.get_file(
    origin="https://huggingface.co/datasets/cnn_dailymail/resolve/main/data/dailymail_stories.tgz",
    extract=True
)
```

NOTE: **Directory structure of the extracted cnn and dailymail folders👇**
```
cnn
  stories
    438411e10e1ef79b47cc48cd95296d85798c1e38.story
    e453e379e8a70af2d3dff1c75c41b0a35edbe9cc.story
    2079f35aca44978a7985afe0ddacdf02bedf98f2.story
    4702f28c198223157bb8f69665b039d560eebb0f.story
    db3e2ea79323a98379228b17cd3b9dec17dbd2cb.story
    ...
    ...
    ...

dailymail
  stories
    f4ba18635997139c751311b9f2ad18f455dd7c98.story
    4a3ef32cff589c85ad0d22724e2ed747c0dacf87.story
    5375ed75939108c72001b043d3b4799c47f32be9.story
    fe9e57c21e21fb4ec26e394f0e92824f38d18a95.story
    6a544b5cdd2384be6cc657b265d7aa2de72a99e0.story
    ...
    ...
    ...

```

**Step 5: Copy the stories of CNN and Dailymail to `data/cnn` and `data/dailymail` dir**

```python
import shutil

# Source directory (files in Colab environment)
cnn_source_dir = '/root/.keras/datasets/cnn/stories'  # Replace with the path to your dataset in Colab
dailymail_source_dir = '/root/.keras/datasets/dailymail/stories'

# Destination directory (cloned repository directory)
cnn_destination_dir = os.path.join(repo_dir, "data/cnn")
dailymail_destination_dir = os.path.join(repo_dir, "data/dailymail")

# Copy the dataset files to the cloned repository directory
shutil.copytree(cnn_source_dir, cnn_destination_dir)
shutil.copytree(dailymail_source_dir, dailymail_destination_dir)
```

**Step 6: Commit the Changes in GitHub**
```python
repo = Repo(repo_dir)
index = repo.index
index.add([path_to_added_file])
author = Actor("<USERNAME>", "yourexample@gmail.com")

index.commit("Added cnn and dailymail stories in /data folder.", author=author)

# Change current working directory to repo directory
os.chdir(repo_dir)

# Change the remote Origin
!git remote remove origin
!git remote add origin https://<USERNAME>:<PERSONALACCESTOKEN>@github.com/<USERNAME>/<REPO_NAME>.git

# Push the changes
!git push origin main
```

---

**MORE TO BE ADDED IN THE FUTURE**
