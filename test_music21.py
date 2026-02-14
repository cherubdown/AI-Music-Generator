from music21 import corpus, converter

def test_music21():
    print("Loading chorales from music21...")
    chorales = corpus.chorales.Iterator(returnType='filename')
    
    count = 0
    for chorale in chorales:
        count += 1
        if count <= 5:
            print(f"Found: {chorale}")
            c = converter.parse(chorale)
            print(f"Parts: {len(c.parts)}")
            
    print(f"Total chorales found (first 5 shown): {count}") # Iterate all to count? No, just show a few.

if __name__ == "__main__":
    test_music21()
