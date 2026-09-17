#!/usr/bin/env python3
"""Culture-100 benchmark fact table (FROZEN 2026-09-16, audited 8 batches).

Schema: fact_id, domain, question, accepted_answer, aliases, difficulty,
template, source, source_url, verified_date.
Amendments baked in: GAME-008 +EA Canada; HIST-004 +Columbia/Eagle;
HIST-010 +Herculaneum.

Usage: python3 scripts/build_culture100.py [--out evals/failure_analysis_362289/culture_100.jsonl]
"""
import json, argparse

F = []  # (id, domain, question, answer, aliases, difficulty, template, source)


def add(fid, dom, q, a, al, diff, tmpl, src="audit"):
    F.append({"fact_id": fid, "domain": dom, "question": q,
              "accepted_answer": a, "aliases": al, "difficulty": diff,
              "template": tmpl, "source": src, "source_url": "",
              "verified_date": "2026-09-16"})


# ── Batch 1: Music (20) ──
add("MUS-001", "music", "The Beatles were a band from which city?", "Liverpool", ["Liverpool, England"], "easy", "WH", "Beatles archive")
add("MUS-002", "music", "The Beatles' second UK album was titled what?", "With The Beatles", ["With The Beatles"], "medium", "WH", "Beatles archive")
add("MUS-003", "music", "Complete: The Beatles' first UK single was Love Me ___.", "Do", ["Love Me Do"], "easy", "cloze", "Beatles archive")
add("MUS-004", "music", "Who was the drummer of The Beatles for most of their famous period?", "Ringo Starr", ["Richard Starkey", "Ringo"], "easy", "WH", "Beatles archive")
add("MUS-005", "music", "Which Beatles member played bass?", "Paul McCartney", ["Paul"], "easy", "WH", "Beatles archive")
add("MUS-006", "music", "The Beatles' final public performance took place on the rooftop of which building?", "Apple Corps headquarters at 3 Savile Row", ["3 Savile Row", "Apple Corps building"], "hard", "WH", "Beatles archive")
add("MUS-007", "music", "In which decade did The Beatles break up?", "1970s", ["the 1970s", "1970"], "easy", "WH", "Beatles archive")
add("MUS-008", "music", "\"All You Need Is Love\" was performed by which band during the 1967 Our World broadcast?", "The Beatles", ["Beatles"], "easy", "WH", "Beatles archive")
add("MUS-009", "music", "Complete: The Beatles' album released in November 1963 was With The ___.", "Beatles", ["With The Beatles"], "medium", "cloze", "Beatles archive")
add("MUS-010", "music", "George Harrison was a member of which band?", "The Beatles", ["Beatles"], "easy", "WH", "Beatles archive")
add("MUS-011", "music", "Linkin Park is an American rock band formed in what year?", "1996", ["1996 CE"], "medium", "WH", "Wikipedia/Linkin Park")
add("MUS-012", "music", "Chester Bennington was the lead vocalist of which band?", "Linkin Park", ["Linkin Park"], "easy", "WH", "Wikipedia/Chester Bennington")
add("MUS-013", "music", "Complete: \"In the End\" is a song by ___.", "Linkin Park", ["Linkin Park"], "easy", "cloze", "Wikipedia/In the End")
add("MUS-014", "music", "Which American rock band released \"In the End\"?", "Linkin Park", ["Linkin Park"], "easy", "WH", "Wikipedia/In the End")
add("MUS-015", "music", "Linkin Park's lead vocalist Chester Bennington was born in which U.S. state?", "Arizona", ["Arizona", "AZ"], "hard", "WH", "Wikipedia/Chester Bennington")
add("MUS-016", "music", "What instrument is Paul McCartney particularly associated with in The Beatles?", "bass guitar", ["bass", "bass guitar"], "easy", "WH", "Beatles archive")
add("MUS-017", "music", "Which Beatles album contains \"Tomorrow Never Knows\"?", "Revolver", ["Revolver"], "medium", "WH", "Beatles archive")
add("MUS-018", "music", "Which Beatles album is associated with the song \"Strawberry Fields Forever\"?", "Magical Mystery Tour", ["Magical Mystery Tour"], "medium", "WH", "Beatles archive")
add("MUS-019", "music", "Complete: The Beatles were formed in ___, England.", "Liverpool", ["Liverpool, England"], "easy", "cloze", "Beatles archive")
add("MUS-020", "music", "The Beatles' official history describes their early development in Liverpool and Hamburg. Which German city was central to their early live career?", "Hamburg", ["Hamburg, Germany"], "medium", "WH", "Beatles archive")

# ── Batch 2: Film/TV (20) ──
add("FILM-001", "film", "Who directed Titanic (1997)?", "James Cameron", ["James Cameron"], "easy", "WH", "Paramount")
add("FILM-002", "film", "Titanic (1997) stars Leonardo DiCaprio and which actress as Rose?", "Kate Winslet", ["Kate Winslet"], "easy", "WH", "Paramount")
add("FILM-003", "film", "Complete: Titanic (1997) was directed by James ___.", "Cameron", ["James Cameron"], "easy", "cloze", "Paramount")
add("FILM-004", "film", "In Titanic, Leonardo DiCaprio played which character?", "Jack Dawson", ["Jack"], "easy", "WH", "Paramount")
add("FILM-005", "film", "In Titanic, Kate Winslet played which character?", "Rose DeWitt Bukater", ["Rose", "Rose DeWitt Bukater"], "easy", "WH", "Paramount/RT")
add("FILM-006", "film", "The Dark Knight was directed by whom?", "Christopher Nolan", ["Christopher Nolan"], "easy", "WH", "audit")
add("FILM-007", "film", "Complete: Christian Bale played Bruce Wayne/Batman in The Dark ___.", "Knight", ["The Dark Knight"], "easy", "cloze", "audit")
add("FILM-008", "film", "Who played the Joker in The Dark Knight (2008)?", "Heath Ledger", ["Heath Ledger"], "easy", "WH", "audit")
add("FILM-009", "film", "The Dark Knight is a film in which superhero franchise?", "Batman", ["Batman", "DC", "Batman", "DC/Batman"], "medium", "WH", "audit")
add("FILM-010", "film", "Which 1999 film follows Neo, a hacker who discovers the nature of his reality?", "The Matrix", ["The Matrix"], "medium", "WH", "audit")
add("FILM-011", "film", "Who directed The Matrix (1999)?", "The Wachowskis", ["Lana Wachowski and Lilly Wachowski", "Wachowskis"], "medium", "WH", "audit")
add("FILM-012", "film", "Complete: The protagonist of The Matrix is commonly known as ___.", "Neo", ["Thomas Anderson", "Neo"], "easy", "cloze", "audit")
add("FILM-013", "film", "Which animated film features the characters Woody and Buzz Lightyear?", "Toy Story", ["Toy Story"], "easy", "WH", "Disney")
add("FILM-014", "film", "Toy Story was produced by which animation studio?", "Pixar", ["Pixar Animation Studios"], "easy", "WH", "Wikipedia/Toy Story")
add("FILM-015", "film", "Complete: Toy Story was released in ___.", "1995", ["1995"], "easy", "cloze", "Disney")
add("FILM-016", "film", "Which television series follows the fictional Simpson family in Springfield?", "The Simpsons", ["The Simpsons"], "easy", "WH", "audit")
add("FILM-017", "film", "What is the first name of the father in The Simpsons?", "Homer", ["Homer Simpson"], "easy", "WH", "audit")
add("FILM-018", "film", "Breaking Bad primarily follows which chemistry teacher who becomes involved in producing methamphetamine?", "Walter White", ["Walter White"], "medium", "WH", "audit")
add("FILM-019", "film", "Complete: Bryan Cranston played Walter ___ in Breaking Bad.", "White", ["Walter White"], "easy", "cloze", "audit")
add("FILM-020", "film", "Which television series is set primarily in the fictional town of Hawkins, Indiana?", "Stranger Things", ["Stranger Things"], "easy", "WH", "audit")

# ── Batch 3: Literature (15) ──
add("LIT-001", "literature", "Who wrote Pride and Prejudice?", "Jane Austen", ["Austen", "Jane Austen"], "easy", "WH", "ERIC")
add("LIT-002", "literature", "Complete: Pride and ___ was written by Jane Austen.", "Prejudice", ["Pride and Prejudice"], "easy", "cloze", "ERIC")
add("LIT-003", "literature", "Which novel features Elizabeth Bennet as its protagonist?", "Pride and Prejudice", ["Pride and Prejudice"], "easy", "WH", "ERIC")
add("LIT-004", "literature", "Who wrote Great Expectations?", "Charles Dickens", ["Dickens", "Charles Dickens"], "easy", "WH", "ERIC")
add("LIT-005", "literature", "Complete: Great Expectations was written by Charles ___.", "Dickens", ["Charles Dickens"], "easy", "cloze", "ERIC")
add("LIT-006", "literature", "Which Shakespeare tragedy features the character Hamlet, Prince of Denmark?", "Hamlet", ["Hamlet"], "easy", "WH", "Folger/RSC")
add("LIT-007", "literature", "Who wrote Hamlet?", "William Shakespeare", ["Shakespeare", "William Shakespeare"], "easy", "WH", "Folger/RSC")
add("LIT-008", "literature", "Complete: William Shakespeare wrote the tragedy ___.", "Hamlet", ["Hamlet"], "easy", "cloze", "Folger/RSC")
add("LIT-009", "literature", "Which Shakespeare play features the characters Romeo and Juliet?", "Romeo and Juliet", ["Romeo and Juliet"], "easy", "WH", "Folger/RSC")
add("LIT-010", "literature", "Who wrote The Odyssey?", "Homer", ["Homer"], "medium", "WH", "ERIC")
add("LIT-011", "literature", "Complete: Homer's two major epic poems include The Iliad and The ___.", "Odyssey", ["The Odyssey"], "easy", "cloze", "ERIC")
add("LIT-012", "literature", "Who wrote Frankenstein?", "Mary Shelley", ["Shelley", "Mary Shelley"], "easy", "WH", "stylometry/UVA")
add("LIT-013", "literature", "Which novel was written by Mary Shelley and first published in 1818?", "Frankenstein", ["Frankenstein"], "medium", "WH", "UVA anthology")
add("LIT-014", "literature", "Who wrote The Adventures of Huckleberry Finn?", "Mark Twain", ["Samuel Clemens", "Mark Twain"], "medium", "WH", "ERIC")
add("LIT-015", "literature", "Complete: Paradise Lost was written by John ___.", "Milton", ["John Milton"], "medium", "cloze", "Durham/UVA")

# ── Batch 4: Art (10) ──
add("ART-001", "art", "Who painted the Mona Lisa?", "Leonardo da Vinci", ["Leonardo", "Leonardo da Vinci"], "easy", "WH", "audit")
add("ART-002", "art", "Complete: Leonardo da Vinci painted the Mona ___.", "Lisa", ["Mona Lisa"], "easy", "cloze", "audit")
add("ART-003", "art", "Which artist painted The Starry Night?", "Vincent van Gogh", ["Van Gogh", "Vincent van Gogh"], "easy", "WH", "audit")
add("ART-004", "art", "The Starry Night was painted by which Dutch artist?", "Vincent van Gogh", ["Van Gogh", "Vincent van Gogh"], "easy", "WH", "audit")
add("ART-005", "art", "Who painted The Persistence of Memory?", "Salvador Dali", ["Dali", "Salvador Dali", "Salvador Dalí", "Dalí"], "medium", "WH", "MoMA")
add("ART-006", "art", "Complete: The Persistence of Memory is a painting by Salvador ___.", "Dali", ["Salvador Dali", "Dali", "Salvador Dalí", "Dalí"], "medium", "cloze", "MoMA")
add("ART-007", "art", "Which artistic movement is Pablo Picasso strongly associated with as a pioneer?", "Cubism", ["Cubist movement", "Cubism"], "medium", "WH", "Smarthistory")
add("ART-008", "art", "Who painted Guernica?", "Pablo Picasso", ["Picasso", "Pablo Picasso"], "easy", "WH", "Britannica")
add("ART-009", "art", "Complete: Picasso's Guernica responds to the bombing of the Spanish town of ___.", "Guernica", ["Guernica"], "medium", "cloze", "Britannica")
add("ART-010", "art", "Which artist created the sculpture The Thinker?", "Auguste Rodin", ["Rodin", "Auguste Rodin"], "easy", "WH", "audit")

# ── Batch 5: Games (10, GAME-008 amended +EA Canada) ──
add("GAME-001", "games", "Which company developed Minecraft?", "Mojang Studios", ["Mojang", "Mojang Studios"], "easy", "WH", "audit")
add("GAME-002", "games", "Complete: Minecraft was originally created by Markus ___.", "Persson", ["Notch", "Markus Persson"], "easy", "cloze", "audit")
add("GAME-003", "games", "Who is commonly known by the nickname \"Notch\" and originally created Minecraft?", "Markus Persson", ["Markus Persson", "Notch"], "easy", "WH", "audit")
add("GAME-004", "games", "Which company published Grand Theft Auto V?", "Rockstar Games", ["Rockstar"], "easy", "WH", "Take-Two/Wiki")
add("GAME-005", "games", "Grand Theft Auto V was developed primarily by which Rockstar studio?", "Rockstar North", ["Rockstar North"], "medium", "WH", "Take-Two/Wiki")
add("GAME-006", "games", "Complete: The Legend of Zelda is a Nintendo video game ___.", "series", ["Nintendo series", "video game series"], "easy", "cloze", "audit")
add("GAME-007", "games", "Which company developed Portal 2?", "Valve", ["Valve Corporation"], "easy", "WH", "audit")
add("GAME-008", "games", "Need for Speed: Most Wanted (2005) was developed by which studio?", "EA Black Box", ["Black Box", "EA Black Box", "EA Canada"], "medium", "WH", "Wiki/MobyGames/GI.biz")
add("GAME-009", "games", "Complete: Need for Speed: Most Wanted (2005) is part of the ___ for Speed series.", "Need for Speed", ["Need for Speed"], "easy", "cloze", "Wiki")
add("GAME-010", "games", "Which studio developed Half-Life 2?", "Valve", ["Valve Corporation"], "easy", "WH", "audit")

# ── Batch 6: Historical culture (10, HIST-004 +Columbia/Eagle, HIST-010 +Herculaneum) ──
add("HIST-001", "history", "In which year did the Apollo 11 Moon landing occur?", "1969", ["1969 CE"], "easy", "WH", "NASA")
add("HIST-002", "history", "Who was the first person to walk on the Moon?", "Neil Armstrong", ["Armstrong", "Neil Armstrong"], "easy", "WH", "NASA")
add("HIST-003", "history", "Complete: Apollo 11 landed on the Moon in ___.", "1969", ["1969 CE"], "easy", "cloze", "NASA/Smithsonian")
add("HIST-004", "history", "Which spacecraft carried the Apollo 11 astronauts to the Moon?", "Apollo 11", ["Apollo 11 mission", "Columbia", "Eagle"], "easy", "WH", "NASA/Smithsonian")
add("HIST-005", "history", "In which year did the Berlin Wall fall?", "1989", ["1989 CE"], "easy", "WH", "Wiki")
add("HIST-006", "history", "Complete: The Berlin Wall fell in ___.", "1989", ["1989 CE"], "easy", "cloze", "Wiki")
add("HIST-007", "history", "Who was the first emperor of the Roman Empire?", "Augustus", ["Augustus Caesar", "Octavian"], "medium", "WH", "audit")
add("HIST-008", "history", "Complete: The Magna Carta was sealed in England in ___.", "1215", ["1215 CE"], "medium", "cloze", "National Archives UK")
add("HIST-009", "history", "Which ancient civilization built Machu Picchu?", "Inca", ["Inca civilization", "Inca Empire"], "easy", "WH", "Wiki/UNESCO")
add("HIST-010", "history", "Which city was buried by the eruption of Mount Vesuvius in 79 CE?", "Pompeii", ["Pompeii", "Pompeii (ancient city)", "Herculaneum"], "easy", "WH", "Wiki")

# ── Batch 7: Geography/culture (10) ──
add("GEO-001", "geography", "What is the capital of Japan?", "Tokyo", ["Tokyo"], "easy", "WH", "audit")
add("GEO-002", "geography", "Complete: Mount Fuji is located in ___.", "Japan", ["Japan"], "easy", "cloze", "audit")
add("GEO-003", "geography", "Which country is home to the ancient city of Petra?", "Jordan", ["Jordan"], "easy", "WH", "UNESCO")
add("GEO-004", "geography", "Petra is located in which modern-day country?", "Jordan", ["Jordan"], "easy", "WH", "UNESCO/Wiki")
add("GEO-005", "geography", "Which city is known for the Eiffel Tower?", "Paris", ["Paris", "France"], "easy", "WH", "audit")
add("GEO-006", "geography", "Complete: The Eiffel Tower is located in ___.", "Paris", ["Paris", "Paris, France"], "easy", "cloze", "audit")
add("GEO-007", "geography", "Which country contains the Great Barrier Reef?", "Australia", ["Australia"], "easy", "WH", "audit")
add("GEO-008", "geography", "Complete: The Great Barrier Reef lies off the coast of ___.", "Australia", ["Australia"], "easy", "cloze", "audit")
add("GEO-009", "geography", "Which country is home to the Taj Mahal?", "India", ["India"], "easy", "WH", "audit")
add("GEO-010", "geography", "In which Indian city is the Taj Mahal located?", "Agra", ["Agra", "Uttar Pradesh"], "easy", "WH", "audit")

# ── Batch 8: Misc (5) ──
add("MISC-001", "misc", "What is the largest ocean on Earth?", "Pacific Ocean", ["Pacific"], "easy", "WH", "audit")
add("MISC-002", "misc", "The largest ocean on Earth is the ___ Ocean.", "Pacific", ["Pacific Ocean"], "easy", "Completion", "audit")
add("MISC-003", "misc", "Which element has the chemical symbol Fe?", "Iron", [], "easy", "WH", "audit")
add("MISC-004", "misc", "The chemical symbol Fe stands for ___.", "Iron", [], "easy", "Cloze", "audit")
add("MISC-005", "misc", "Which planet is known for its prominent ring system?", "Saturn", [], "easy", "WH", "Wiki/NASA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evals/failure_analysis_362289/culture_100.jsonl")
    args = ap.parse_args()
    assert len(F) == 100, f"expected 100 facts, got {len(F)}"
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for r in F:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"WROTE {len(F)} facts -> {args.out}")
    print("by domain:", dict(Counter(r["domain"] for r in F)))


if __name__ == "__main__":
    main()
