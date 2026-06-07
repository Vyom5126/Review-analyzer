# APPROACH.md - How I Actually Built This 

## Why I built this

I wanted to build something better than a normal sentiment analysis model that only says whether a review is positive or negative.
A review can contain different opinions about different parts of a product. For example, a customer may give a product 3 stars. That rating alone does not tell us much.
The useful part is knowing what the customer liked and disliked. They may like the screen but not like the battery. By finding opinions about each feature separately, companies can understand customer feedback much better and help them improve their products.

---

## The dataset decision

I used Amazon Electronics reviews from the McAuley Lab dataset on Hugging Face.

One thing I realized early on was that I didn't need to manually label the data. The star ratings gave me the sentiment labels. Reviews with 1–2 stars were treated as negative, 4–5 stars as positive, and 3 stars as neutral.
This saved a lot of time and effort.

One practical problem: the full dataset is 22GB. I used Hugging Face's streaming feature to pull only the first 50,000 rows without downloading the whole thing. 

---

## Why I built two sentiment models instead of one

I built a simple model first (TF-IDF + Logistic Regression) before
building the better one (Sentence-Transformers + XGBoost).

The simple model gave me a baseline, a number I could measure improvement against.

The other reason: TF-IDF + Logistic Regression is extremely fast and decent. If the gap between it and the upgraded model had been tiny, I would have stuck with the simple one.

**The class imbalance problem and why it nearly broke the upgrade.**

The dataset is heavily skewed: 80% positive, 14% negative, 6% neutral. The baseline handled this explicitly with `class_weight='balanced'` in Logistic Regression, which automatically up-weights minority classes so the model doesn't just learn to always predict positive.

In my first XGBoost attempt I forgot to apply the same correction. The model learned to predict positive almost exclusively and scored F1=0.02 on the neutral class, dragging the macro F1 to 0.4997 worse than the baseline. This looked like the sentence embeddings weren't working, but the real cause was the missing class balancing.

The fix was passing `sample_weight=compute_sample_weight('balanced', y_train)` to `xgb_model.fit()`. This is the XGBoost equivalent of `class_weight='balanced'`. With that correction, the neutral class F1 jumped from 0.02 to 0.42, and the overall macro F1 reached 0.6767  a +12.7% improvement over the baseline. The lesson: when comparing models, make sure they are solving the same problem under the same conditions before drawing conclusions about which approach is better.

---

## Why sentence transformers over just TF-IDF

TF-IDF works by looking at the words in a review and counting how important they are. However, it does not understand the actual meaning of a sentence.

To solve this problem, I used sentence-transformers. Instead of focusing only on words, they convert each sentence into a set of numbers that represent its meaning. Because of this, sentences with similar meanings are placed close together, even if they use different words.

This helped the model understand reviews better. It became more effective at handling different ways of expressing the same opinion, including synonyms, unusual wording, and some cases of sarcasm.

The main disadvantage is that sentence-transformers require more time and computing power. For my project, this was not a major issue because the application analyzes one review at a time.

---

## Why rule based aspect extraction is actually the right call

To identify product aspects, I created a simple list of keywords for each aspect. For example, words like battery, charge, charging, drain, and life all point to the battery aspect. Similarly, words related to the screen, camera, performance, and price were grouped together.

This approach was quick to implement and worked well for electronics reviews. Most customers use common words when talking about product features, so a small set of keywords was enough to capture the majority of aspect mentions.

Another advantage was that the system was easy to understand and explain. If a review was linked to the battery aspect, I could clearly show which keyword caused the match. This made the results more transparent and easier to debug.

For this project, the simple keyword based approach was an effective solution because it required no additional training data and provided reliable results with very little development time.

---

## The core technical challenge and how I solved it

One of the biggest challenges in this project was handling sentences that talk about multiple aspects at the same time.

For example, a review might say, "The battery is good but the display is bad." In this sentence, the customer is giving a positive opinion about the battery and a negative opinion about the display.

My pipeline splits reviews into sentences, then classifies each sentence for sentiment. But this sentence has two aspects  battery and screen. When I ran the whole sentence through the sentiment model, it saw mixed signals and returned the wrong result for both aspects.

To solve this, I focused only on the words around each aspect keyword. For every aspect, I looked at a small window of words before and after the keyword. In most cases, these nearby words contain the sentiment for that specific aspect. For example, in "battery is good", the word good is close to battery. Similarly, in "display is bad", the word bad is close to display..

I also created a small list of positive and negative words. If positive words appeared near an aspect, I marked it as positive. If negative words appeared, I marked it as negative. If both positive and negative words were present, I treated it as neutral and used the main sentiment model for the final decision.

This simple approach worked well and was more reliable for my project.

---

## One thing that didn't work

I also experimented with a zero shot classification approach using the BART model for aspect detection.

Instead of creating a list of keywords for each aspect, I could give the model aspect names such as battery, screen, or price, and let it decide what the review was talking about.

Although this approach was flexible, I found some practical limitations.

The first issue was speed. The model took around 1–2 seconds to analyze a single review. While this may be acceptable for testing or small applications, it becomes too slow when processing a large number of reviews.

The second issue was accuracy. Sometimes the model assigned aspects to sentences that were not actually related to those aspects. For example, a sentence like "The seller was very helpful" could be incorrectly linked to an aspect even though it was not discussing the product itself.

Because of this, I found that my keyword based approach gave more reliable results for this project. It was much faster and correctly identified most aspect mentions in electronics reviews.

I still kept the BART model as an optional backup in the code. If the keyword based method could not identify any aspect, the review could be passed to BART. However, in most cases, the keyword based approach was enough and provided results much more quickly.

---

## What I would do differently with more time

**Better aspect coverage.** One improvement would be to expand the number of aspects the system can detect. At the moment, it focuses on only a few common aspects such as battery, screen, and price. However, electronics reviews often discuss other topics like build quality, software, customer service, and sound quality. Adding more keywords and aspects would help the system capture a larger portion of customer feedback.

**Fine-tune the sentiment model on electronics reviews specifically.** The current model was trained on general product reviews, but some words can have different meanings in the electronics domain. A model trained on electronics related data could better understand this type of language and provide more accurate predictions.

**Handle negation better.** The system could be improved by handling negation more effectively. For example, in a sentence like "The battery is not bad", the actual meaning is positive even though the word bad appears. The current approach may sometimes miss this and classify the sentiment incorrectly. A simple improvement would be to detect words such as not, never, or doesn't before a sentiment word and adjust the sentiment accordingly.

---

## What I'm most proud of

The main idea behind this project is that an overall sentiment score does not always provide enough information. A review may have a neutral or average rating, but it can still contain valuable feedback about different parts of a product.

For example, a customer may give a phone 3 stars because they like the screen but are unhappy with the battery. Looking only at the overall rating hides these details. By analyzing sentiment for individual aspects, we can better understand what customers like and dislike.

This information can be very useful for businesses. Instead of seeing only an average rating, they can identify the specific features that need improvement and the features that customers already appreciate.

The system is not perfect. It currently covers a limited number of aspects, uses simple rules to identify them, and relies on models trained on general review data. However, it successfully demonstrates the main idea and provides meaningful insights from customer reviews.

Overall, this project shows that breaking reviews into individual aspects can reveal much more useful information than looking at overall sentiment alone, making it a strong foundation for future improvements.