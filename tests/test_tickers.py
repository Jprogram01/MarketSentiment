from marketsentiment.nlp import extract_tickers


def test_cashtags_are_extracted_and_uppercased():
    assert extract_tickers("$aapl and $Nvda ripping 🚀") == ["AAPL", "NVDA"]


def test_cashtags_dedupe_in_first_seen_order():
    assert extract_tickers("$GME $gme $AMC $GME") == ["GME", "AMC"]


def test_bare_tokens_ignored_without_known_set():
    # No cashtag, no known-ticker set -> nothing matches.
    assert extract_tickers("TSLA to the moon") == []


def test_bare_tokens_matched_against_known_set():
    assert extract_tickers("TSLA and NVDA look strong", known={"TSLA", "NVDA"}) == ["TSLA", "NVDA"]


def test_stopwords_filtered_even_if_in_known_set():
    # "DD" is a market slang stopword; it must not be treated as a ticker.
    assert extract_tickers("GME DD is bullish", known={"GME", "DD"}) == ["GME"]


def test_cashtag_wins_over_stopword_filtering():
    # An explicit cashtag is unambiguous even for a stopword-like symbol.
    assert extract_tickers("$AI is the theme", known=None) == ["AI"]
