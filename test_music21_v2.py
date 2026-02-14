from music21 import corpus

def test_music21():
    print("Loading chorales from music21...")
    try:
        c = corpus.parse('bach/bwv269')
        print("Success: bach/bwv269")
        print(f"Parts: {len(c.parts)}")
    except Exception as e:
        print(f"Failed 'bach/bwv269': {e}")
        
    try:
        c = corpus.parse('bwv269')
        print("Success: bwv269")
    except Exception as e:
        print(f"Failed 'bwv269': {e}")

if __name__ == "__main__":
    test_music21()
