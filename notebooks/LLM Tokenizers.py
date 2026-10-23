# %% [markdown]
# Source notebook: LLM Tokenizers.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 2: Comparing Trained LLM Tokenizers

# %% [markdown]
# In this notebook of lesson 2, you will work with several tokenizers associated with different LLMs and explore how each tokenizer approaches tokenization differently. 

# %% [markdown]
# ## Setup

# %% [markdown]
# We start with setting up the lab by installing the `transformers` library and ignoring the warnings. The requirements for this lab are already installed, so you don't need to uncomment the following cell.

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> file:</b> If you'd like to access the requirements file: 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

# %%
# Notebook cell 6
# !pip install transformers>=4.46.1

# %%
# Notebook cell 7
# Warning control
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Tokenizing Text

# %% [markdown]
# In this section, you will tokenize the sentence "Hello World!" using the tokenizer of the [`bert-base-cased` model](https://huggingface.co/google-bert/bert-base-cased). 
# 
# Let's import the `Autotokenizer` class, define the sentence to tokenize, and instantiate the tokenizer.

# %% [markdown]
# <p style="background-color:#fff1d7; padding:15px; "> <b>FYI: </b> The transformers library has a set of Auto classes, like AutoConfig, AutoModel, and AutoTokenizer. The Auto classes are designed to automatically do the job for you.</p>

# %%
# Notebook cell 11
from transformers import AutoTokenizer

# %%
# Notebook cell 12
# define the sentence to tokenize
sentence = "Hello world!"

# %%
# Notebook cell 13
# load the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# %% [markdown]
# You'll now apply the tokenizer to the sentence. The tokeziner splits the sentence into tokens and returns the IDs of each token.

# %%
# Notebook cell 15
# apply the tokenizer to the sentence and extract the token ids
token_ids = tokenizer(sentence).input_ids

# %%
# Notebook cell 16
print(token_ids)

# %% [markdown]
# To map each token ID to its corresponding token, you can use the `decode` method of the tokenizer.

# %%
# Notebook cell 18
for id in token_ids:
    print(tokenizer.decode(id))

# %% [markdown]
# ## Visualizing Tokenization
# 
# In this section, you'll wrap the code of the previous section in the function `show_tokens`. The function takes in a text and the model name, and prints the vocabulary length of the tokenizer and a colored list of the tokens. 

# %%
# Notebook cell 20
# A list of colors in RGB for representing the tokens
colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence: str, tokenizer_name: str):
    """ Show the tokens each separated by a different color """

    # Load the tokenizer and tokenize the input
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids

    # Extract vocabulary length
    print(f"Vocab length: {len(tokenizer)}")

    # Print a colored list of tokens
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )

# %% [markdown]
# Here's the text that you'll use to explore the different tokenization strategies of each model.

# %%
# Notebook cell 22
text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""

# %% [markdown]
# You'll now again use the tokenizer of `bert-base-cased` and compare its tokenization strategy to that of `Xenova/gpt-4`

# %% [markdown]
# **bert-base-cased**

# %%
# Notebook cell 25
show_tokens(text, "bert-base-cased")

# %% [markdown]
# **Optional - bert-base-uncased**
# 
# You can also try the uncased version of the bert model, and compare the vocab length and tokenization strategy of the two bert versions.

# %%
# Notebook cell 27
show_tokens(text, "bert-base-uncased")

# %% [markdown]
# **GPT-4**

# %%
# Notebook cell 29
show_tokens(text, "Xenova/gpt-4")

# %% [markdown]
# ### Optional Models to Explore
# 
# You can also explore the tokenization strategy of other models. The following is a suggested list. Make sure to consider the following features when you're doing your comparison:
# - Vocabulary length
# - Special tokens
# - Tokenization of the tabs, special characters and special keywords

# %% [markdown]
# **gpt2**

# %%
# Notebook cell 32
show_tokens(text, "gpt2")

# %% [markdown]
# **Flan-T5-small**

# %%
# Notebook cell 34
show_tokens(text, "google/flan-t5-small")

# %% [markdown]
# **Starcoder 2 - 15B**

# %%
# Notebook cell 36
show_tokens(text, "bigcode/starcoder2-15b")

# %% [markdown]
# **Phi-3**

# %%
# Notebook cell 38
show_tokens(text, "microsoft/Phi-3-mini-4k-instruct")

# %% [markdown]
# **Qwen2 - Vision-Language Model**

# %%
# Notebook cell 40
show_tokens(text, "Qwen/Qwen2-VL-7B-Instruct")

# %% [markdown]
# <p style="background-color:#f2f2ff; padding:15px; border-width:3px; border-color:#e2e2ff; border-style:solid; border-radius:6px"> ⬇
# &nbsp; <b>Download Notebooks:</b> If you'd like to donwload the notebook: 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>. For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
