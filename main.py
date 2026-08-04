from impostors import ImpostorsVerifier
import numpy

def load_text(files): 
    # load the text from a list of files
    texts = []
    for file in files : 
        result = ""
        with open(file, "r", encoding="utf-8") as f:
                for line in f.readlines(): 
                     result = result + line.strip()
        texts.append(result)
    return texts


if __name__ == "__main__":
    # Listing the texts we know the author of
    author_file = [
        "texts/lapopeliniere.txt",
    ]
    
    # Text we want to test
    anonymous_file = ["texts/advertissemens.txt"]
    
    # Control group of texts by other authors (same era/genre style)
    impostors_files = [
        "texts/aubigné.txt",
        "texts/laplanche.txt",
        "texts/goulart.txt",
    ]

    # Load texts from files
    print("Parsing author")
    author_text = load_text(author_file)
    print("Parsing anonymous")
    anonymous_text = load_text(anonymous_file)
    print("Parsing impostors")
    impostors = load_text(impostors_files)

    print("Running impostor method")
    # Impostors verifier
    verifier = ImpostorsVerifier(n_iterations=100, feature_fraction=0.6)
    score = verifier.verify(author_text, anonymous_text, impostors)
    
    print(f"Verification Score: {score:.2%}")
    if score >= 0.70:
        print("Decision: YES (The text matches the author's stylistic fingerprint)")
    else:
        print("Decision: NO (The text does not closely match the author relative to the control pool)")