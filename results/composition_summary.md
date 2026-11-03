# Composition results

Grid: 6 colors x 6 shapes (36 combinations). In sections 1, 2 and 4, ten combinations are held out of every training set and unlabeled pool, and are used only for scoring. In section 3 (in context), each trial draws its own example set and made-up words, and asks for one combination outside the set whose color and shape both appear in it. *cross* = one full row plus one full column of the grid (11 combinations). *confounded* = each color always shown with the same shape.

## 1. Small CNN trained from scratch (JAX, 32 x 32 images)

| Training mix (12,000 images) | Post-training (220 images) | Seeds | Held-out combinations right | Color right | Shape right | Seen combinations right |
|---|---|---|---|---|---|---|
| correlated (95% typical pairs) | none | 5 | 8% ± 3% | 97% | 10% | 82% |
| correlated (95% typical pairs) | correlated | 5 | 14% ± 9% | 98% | 16% | 70% |
| correlated (95% typical pairs) | cross | 5 | 11% ± 2% | 86% | 20% | 69% |
| correlated (95% typical pairs) | all26 | 5 | 17% ± 9% | 98% | 18% | 95% |
| balanced over 26 combinations | none | 5 | 30% ± 11% | 98% | 31% | 98% |
| balanced over 26 combinations | correlated | 5 | 33% ± 8% | 99% | 34% | 99% |
| balanced over 26 combinations | cross | 5 | 25% ± 4% | 97% | 27% | 88% |
| balanced over 26 combinations | all26 | 5 | 37% ± 8% | 99% | 37% | 99% |

## 2. Post-training on a frozen pretrained model (Gemini image embeddings, ridge heads)

| Post-training set | Head | Images | Combinations | Held-out combinations right (20 seeds) | Seeds at 100% | Color right | Shape right | Seen combinations right |
|---|---|---|---|---|---|---|---|---|
| cross | one label per attribute | 110 | 11 | 100% ± 0% | 20/20 | 100% | 100% | 100% |
| random11 | one label per attribute | 110 | 11 | 82% ± 13% | 4/20 | 93% | 89% | 83% |
| confounded | one label per attribute | 108 | 6 | 0% ± 0% | 0/20 | 56% | 44% | 23% |
| cross | one label per combination | 110 | 11 | 0% ± 0% | 0/20 | 68% | 32% | 42% |
| all26 | one label per attribute | 110 | 26 | 100% ± 0% | 20/20 | 100% | 100% | 100% |
| all26_full | one label per attribute | 468 | 26 | 100% ± 0% | 20/20 | 100% | 100% | 100% |
| all26_full | one label per combination | 468 | 26 | 0% ± 0% | 0/20 | 54% | 44% | 100% |

## 3. In context (11 labeled examples in made-up words, one new picture)

| Model | Example set | Word order | Test picture | Exact name | 95% CI |
|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite | cross | color word first | one object | 16/16 | 0.81-1.00 |
| Gemini 3.5 Flash-Lite | cross | shape word first | one object | 16/16 | 0.81-1.00 |
| Gemini 3.5 Flash-Lite | cross | color word first | two objects | 12/12 | 0.76-1.00 |
| Gemini 3.5 Flash-Lite | cross | shape word first | two objects | 12/12 | 0.76-1.00 |
| Gemini 3.5 Flash-Lite | random | color word first | one object | 16/16 | 0.81-1.00 |
| Gemini 3.5 Flash-Lite | random | shape word first | one object | 16/16 | 0.81-1.00 |
| Gemini 3.5 Flash-Lite | confounded | color word first | one object | 14/16 | 0.64-0.97 |
| Gemini 3.5 Flash-Lite | confounded | shape word first | one object | 3/16 | 0.07-0.43 |
| Gemini 3.1 Flash-Lite | cross | color word first | one object | 16/16 | 0.81-1.00 |
| Gemini 3.1 Flash-Lite | cross | shape word first | one object | 15/16 | 0.72-0.99 |
| Gemini 3.1 Flash-Lite | cross | color word first | two objects | 12/12 | 0.76-1.00 |
| Gemini 3.1 Flash-Lite | cross | shape word first | two objects | 12/12 | 0.76-1.00 |
| Gemini 3.1 Flash-Lite | random | color word first | one object | 16/16 | 0.81-1.00 |
| Gemini 3.1 Flash-Lite | random | shape word first | one object | 16/16 | 0.81-1.00 |
| Gemini 3.1 Flash-Lite | confounded | color word first | one object | 8/16 | 0.28-0.72 |
| Gemini 3.1 Flash-Lite | confounded | shape word first | one object | 3/16 | 0.07-0.43 |
| Gemma 4 26B-A4B | cross | color word first | one object | 16/16 | 0.81-1.00 |
| Gemma 4 26B-A4B | cross | shape word first | one object | 16/16 | 0.81-1.00 |
| Gemma 4 26B-A4B | cross | color word first | two objects | 12/12 | 0.76-1.00 |
| Gemma 4 26B-A4B | cross | shape word first | two objects | 12/12 | 0.76-1.00 |
| Gemma 4 26B-A4B | random | color word first | one object | 16/16 | 0.81-1.00 |
| Gemma 4 26B-A4B | random | shape word first | one object | 16/16 | 0.81-1.00 |
| Gemma 4 26B-A4B | confounded | color word first | one object | 15/16 | 0.72-0.99 |
| Gemma 4 26B-A4B | confounded | shape word first | one object | 0/16 | 0.00-0.19 |

Confounded examples, all three models: 95 of 96 answers match one of the two readings the examples allow; 79 read the first word as the color and 16 read the second word as the color, in both word orders.
Two-object pictures: 0 of 72 answers used a word belonging to the other object.

## 4. Pilot: in-context labels as post-training data

Labeled set: the 6 confounded combinations (108 images). Unlabeled pool: one image of each of the other 20 shown combinations. Teacher: Gemini 3.5 Flash-Lite with 11 labeled confounded examples in context. Student: ridge heads on Gemini image embeddings. Scored on 160 images of the 10 held-out combinations.

| Post-training data | Teacher labels right | Held-out combinations right | Color and shape swapped |
|---|---|---|---|
| confounded only | - | 0% | 0% |
| + teacher labels, color word first | 16/20 | 100% | 0% |
| + teacher labels, shape word first | 5/20 | 1% | 92% |
| + true labels for the same images | 20/20 | 100% | 0% |
