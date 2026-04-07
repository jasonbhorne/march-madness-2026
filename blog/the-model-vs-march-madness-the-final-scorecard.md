# The Model vs. March Madness: The Final Scorecard

*Post 7 of 7 | April 7, 2026*

Michigan won the national championship Monday night. The model picked Michigan. This is the part where I describe that as a triumph, and also explain the seventeen asterisks.

Michigan beat UConn 69-63 at Lucas Oil Stadium in Indianapolis. Elliot Cadeau led all scorers with 19 points. Trey McKenney hit a step-back three with 1:50 left that pushed the lead to nine and took most of the suspense out of the building. Solo Ball banked in a three with 37 seconds remaining to make it 67-63, and Michigan converted its free throws. Final: Michigan 69, UConn 63. First Michigan title since 1989, first Big Ten title in 26 years.

Braylon Mullins, the man whose buzzer-beater eliminated Duke in the Elite Eight, finished 4-of-17 from the floor. Alex Karaban had 17 points and 11 rebounds and played well enough to deserve a better outcome. Tarris Reed Jr., a former Michigan player, led UConn with 14 rebounds and watched his old program cut down the nets. The basketball gods have a sense of narrative.

UConn's foul trouble was decisive. Solo Ball and Keshaun Demary played a combined 17 minutes in the first half after picking up early fouls, and the Huskies never fully recovered. Michigan exploited it: 32-22 advantage in points in the paint, 15-of-16 at the free throw line. Daniel Lendeborg contributed 13 points while playing through a sprained MCL and a rolled ankle he described at halftime as "super weak." He played anyway. That is not a variable the model tracks.

---

## The Final Scorecard

| Round | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Round of 64 | 22 | 32 | 68.8% |
| Round of 32 | 11 | 16 | 68.8% |
| Sweet 16 | 5 | 8 | 62.5% |
| Elite Eight | 2 | 4 | 50.0% |
| Championship | 1 | 1 | 100.0% |
| **Total** | **41** | **61** | **67.2%** |

The model finished 41-for-61, or 67.2%. Better than a coin flip. Worse than most ESPN bracket challenges. Exactly what you would expect from a system that knows everything about adjusted efficiency margins and nothing about Solo Ball's foul situation.

---

## What the model got right

The early rounds. Twenty-two of thirty-two in the Round of 64 is respectable. Eleven of sixteen in the Round of 32 is consistent. The model identified Michigan as the class of the field before the tournament started and never wavered. It correctly picked Arizona through to the Final Four. It called UConn over Michigan State in the Sweet 16 at 62.1% confidence, which looks sharper in retrospect than it did at the time.

The re-run championship pick was Michigan at 72.5%, with a projected score of Michigan 78, UConn 72. Michigan won 69-63. The model projected a six-point margin. It got a six-point margin. The final score was off, the projected pace was off, and the margin was exactly right. I have no satisfying explanation for this.

---

## What the model got wrong

Duke. Duke was the original championship pick at 57% confidence. Duke lost to UConn in the Elite Eight 73-72 on a Braylon Mullins buzzer-beater. This is the moment the bracket ended and the postmortem began.

The Elite Eight was the worst round: two of four, fifty percent, which is what you get from a coin toss. Houston, picked to the Final Four, lost to Illinois in the Sweet 16. Iowa went to the Elite Eight as a nine-seed. Tennessee made the Elite Eight as a six. The model had Iowa State and Nebraska advancing where Iowa and Tennessee went.

The accuracy curve is honest about the trajectory: 68.8%, 68.8%, 62.5%, 50.0%. The model degraded as the field improved. This is not a bug. It is a structural property of efficiency-based models. When you remove the weak teams, you are left with games that the underlying data genuinely cannot separate, because those teams are genuinely close.

UConn made this specific problem worse than usual. The Huskies beat the model's picks twice before the final. They eliminated Duke when the model had Duke advancing. They beat Illinois in the Final Four when the model's original bracket had put Duke in that exact slot. The model was 0-for-2 against UConn entering Monday night, which is why the re-run result felt less like a prediction and more like a guess that happened to land.

---

## What a CSV cannot capture

The most interesting thing about the 2026 tournament is not what the model got right. It is what the model had no mechanism to see.

UConn's run was built on defensive identity and tournament experience. They had been here before. They knew what close games in March felt like. Braylon Mullins had hit a shot like that before. The model does not have a feature for "has done this before." It has adjusted defensive efficiency. Those are not the same thing.

Foul trouble decided the championship. Ball and Demary were compromised before halftime. Nobody predicts that. The model does not predict that. But it is why UConn lost, and it is the kind of thing that happens in a single game in ways that have nothing to do with season-long ratings.

Lendeborg played a national championship game on a sprained MCL. The model knows his rebounding percentage. It does not know about his ankle, or his halftime assessment of his own condition, or whatever he told himself to keep playing. That gap, between the numbers and the person producing them, is where March Madness actually lives.

This is not an argument against the model. The model was right. But it was right about the outcome while being wrong about most of the mechanisms. That is the epistemological position you end up in when you predict a tournament with historical efficiency data: correct answers for incomplete reasons.

---

## Was it worth it

Yes. The model went 41-for-61 and identified four of the six true contenders before tip-off. It got the champion right, including the re-run after its original pick was eliminated by a buzzer-beater. It correctly mapped where efficiency ratings have signal (early rounds, large talent gaps) and where they break down (late rounds, one-possession games, foul trouble, banked threes with 37 seconds left).

The plan for next year is to add tournament-experience features, late-season trajectory weighting, and coach-specific performance metrics. The model will probably still pick the wrong original champion. That is the appropriate lesson here. The model is a guide to the bracket, not a transcript of it. March exists specifically to remind you of that distinction, usually at the worst possible moment.

Michigan 69, UConn 63. Final record: 41-for-61. See you in 2027.

---

*Series: [Post 1: I Let a Machine Fill Out My Bracket](../i-let-a-machine-fill-out-my-bracket) | [Post 2: Round 1 Results](../round-1-results) | [Post 3: Sweet 16 Predictions](../sweet-16-predictions) | [Post 4: The Bracket Breaks](../the-bracket-breaks) | [Post 5: One Game Left](../one-game-left) | Post 6: The Final Scorecard*
