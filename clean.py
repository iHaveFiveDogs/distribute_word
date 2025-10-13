import re

with open("units.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Replace non-breaking spaces, curly quotes, em/en dashes
text = text.replace("\u00A0", " ")
text = text.replace("\u2013", "-").replace("\u2014", "--")
text = re.sub(r"[“”]", '"', text)
text = re.sub(r"[‘’]", "'", text)

with open("units_clean.txt", "w", encoding="utf-8") as f:
    f.write(text)