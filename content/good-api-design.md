Title: User experience design for APIs
Date: 2017-11-21
Category: Essays
Author: Francois Chollet


Writing code is rarely just a private affair between you and your computer. Code is not just meant for machines; it has human users. It is meant to be read by people, used by other developers, maintained and built upon. Developers who produce better code, in greater quantity, when they are kept happy and productive, working with tools they love. Developers who unfortunately are often being let down by their tools, and left cursing at obscure error messages, wondering why that stupid library doesn't do what they thought it would. Our tools have great potential to cause us pain, especially in a field as complex as software engineering.

User experience (UX) should be central in application programming interface (API) design. A well-designed API, making complicated tasks feel easy, will probably prevent a lot more pain in this world than a brilliant new design for a bedside lamp ever would. So why does API UX design so often feel like an afterthought, compared to even furniture design? Why is there a profound lack of design culture among developers?

![keep the user in mind](/img/api-design/api-design-stock-photo.jpg)

---

Part of it is simply empathic distance. While you're writing code alone in front of your computer, future users are a distant thought, an abstract notion. It's only when you start sitting down next to your users and watch them struggle with your API that you start to realize that UX matters. And, let's face it, most API developers never do that.

Another problem is what I would call "smart engineer syndrome". Programmers tend to assume that end users have sufficient background and context -- because themselves do. But in fact, end users know a tiny fraction of what you know about your own API and its implementation. Besides, smart engineers tend to overcomplicate what they build, because they can easily handle complexity. If you aren't exceptionally bright, or if you are impatient, that fact puts a hard limit on how complicated your software can be -- past a certain level, you simply won't be able to get it to work, so you'll just quit and start over with a cleaner approach. But smart, patient people? They can just deal with the complexity, and they build increasingly ugly Frankenstein monsters, that somehow still walk. This results in the worst kind of API.

One last issue is that some developers force themselves to stick with user-hostile tools, because they perceive the extra difficulty as a badge of honor, and consider thoughtfully-designed tools to be "for the n00bs". This is an attitude I see a lot in the more toxic parts of the deep learning community, where most things tend to be fashion-driven and superficial. But ultimately, this masochistic posturing is self-defeating. In the long run, good design wins, because it makes its adepts more productive and more impactful, thus spreading faster than user-hostile undesign. Good design is infectious.

Like most things, API design is not complicated, it just involves following a few basic rules. They all derive from a founding principle: **you should care about your users.** All of them. Not just the smart ones, not just the experts. Keep the user in focus at all times. Yes, including those befuddled first-time users with limited context and little patience. **Every design decision should be made with the user in mind.**

Here are my three rules for API design.

---

## 1 - Deliberately design end-to-end user workflows.  

Most API developers focus on atomic methods rather than holistic workflows. They let users figure out end-to-end workflows through evolutionary happenstance, given the basic primitives they provided. The resulting user experience is often one long chain of hacks that route around technical constraints that were invisible at the level of individual methods. 

To avoid this, start by listing the most common workflows that your API will be involved in. The use cases that most people will care about. Actually go through them yourself, and take notes. Better yet: watch a new user go through them, and identify pain points. Ruthlessly iron out those pain points. In particular:

- **Your workflows should closely map to domain-specific notions that users care about.** If you are designing an API for cooking burgers, it should probably feature unsurprising objects such as "patty", "cheese", "bun", "grill", etc. And if you are designing a deep learning API, then your core data structures and their methods should closely map to the concepts used by people familiar with the field: models/networks, layers, activations, optimizers, losses, epochs, etc.
- **Ideally, no API element should deal with implementation details.** You do not want the average user to deal with "primary_frame_fn", "defaultGradeLevel", "graph_hook", "shardedVariableFactory", or "hash_scope", because these are not concepts from the underlying problem domain, they are highly specific concepts that come from your internal implementation choices.
- **Deliberately design the user onboarding process.** How are complete newcomers going to find out the best way to solve their use case with your tool? Have an answer ready. Make sure your onboarding material closely maps to what your users care about: *don't teach newcomers how your API is implemented, teach them how they can use it to solve their own problems.*

---

## 2 - Reduce cognitive load for your users.

In the end-to-end workflows you design, always strive to reduce the mental effort that your users have to invest to understand and remember how things work. The less effort and focus you require from your users, the more they can invest in solving their actual problems -- instead of trying to figure out how to use this or that method. In particular:

- **Use consistent naming and code patterns.** Your API naming conventions should be internally consistent (If you usually denote counts via the `num_*` prefix, don't switch to `n_*` in some places), but also consistent with widely recognized external standards. For instance, if you are designing an API for numerical computation in Python, it should not glaringly clash with the Numpy API, which everyone uses. A user-hostile API would arbitrarily use `keepdim` where Numpy uses `keepdims`, would use `dim` where Numpy uses `axis`, etc. And an especially poorly-designed API would just randomly alternate between `axis`, `dim`, `dims`, `axes`, `axis_i`, `dim_i`, for the same concept.
- **Introduce as few new concepts as possible.** It's not just that additional data structures require more effort in order to learn about their methods and properties, it's that they multiply the number of **mental models** that are necessary to grok your API. Ideally, you should only need a single universal mental model from which everything flows (in Keras, that's the `Layer`/`Model`). Definitely avoid having more than 2-3 mental models underlying your workflows.
- **Strike a balance between the number of different classes/functions you have, and the parameterization of these classes/functions.** Having a different class/function for every user action induces high cognitive load, but so does parameter proliferation -- you don't want 35 keyword arguments in a class constructor. Such a balance can be achieved by making your data structures modular and composable.
- **Automate what can be automated.** Strive to reduce the number of user actions required in your workflows. Identify often-repeated code blocks in user-written code, and provide utilities to abstract them away. For instance, in a deep learning API, you should provide automated shape inference instead of requiring users to do mental math to compute expected input shapes in all of their layers.
- **Have clear documentation, with lots of examples.** The best way to communicate to the user how to solve a problem is not to talk about the solution, it is to *show* the solution. Make sure to have concise and readable code examples available for every feature in your API.

**The litmus test I use to tell whether an API is well-designed is the following:** if a new user goes through the workflow for their use case on day one (following the  documentation or a tutorial), and they come back the next day to solve the same problem in a slightly different context, will they be able to follow their workflow *without looking up the documentation/tutorial*? Will they be able to remember their workflow in one shot? *A good API is one where the cognitive load of most workflows is so low that it can be learned in one shot.*

This litmus test also gives you a way to quantify how good or bad an API is, by counting the number of times the average user needs to look up information about a workflow in order to master it. The worst workflows are those that can never be fully remembered, and require following a lengthy tutorial every single time.

---

## 3 - Provide helpful feedback to your users.

Good design is interactive. It should be possible to use a good API while only minimally relying on documentation and tutorials -- by simply trying things that seem intuitive, and acting upon the feedback you get back from the API. In particular:

- **Catch user errors early and anticipate common mistakes.** Do user input validation as soon as possible. Actively keep track of common mistakes that people make, and either solve them by simplifying your API, adding targeted error messages for these mistakes, or having a "solutions to common issues" page in your docs.
- **Have a place where users can ask questions.** How else are you going to keep track of existing pain points you need to fix?
- **Provide detailed feedback messages upon user error.** A good error message should answer: *what happened, in what context? What did the software expect? How can the user fix it?* They should be contextual, informative, and actionable. Every error message that transparently provides the user with the solution to their problem means one less support ticket, multiplied by how many times users run into the same issue.

<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">i want to murder whoever is responsible for this incredibly descriptive error:<br><br>Unable to determine [Unknown Property]. Please specify an [Unknown Property]</p>&mdash; loren schmidt (@lorenschmidt) <a href="https://twitter.com/lorenschmidt/status/929083642919444480?ref_src=twsrc%5Etfw">November 10, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<br>

**For example:**

- In Python, the following would be an extremely bad error message:

```python
AssertionError: '1 != 3'
```

(in general, always use `ValueError` and avoid `assert`).

- Also bad:

```python
ValueError: 'Invalid target shape (600, 1).'
```

- The following is better, but still not sufficient, because it does not tell the user *what they passed*, and does not quite say *how to fix it*:

```python
ValueError: 'categorical_crossentropy requires target.shape[1] == classes'
```


- Now, here's a good example, that says what was passed, what was expected, and how to fix the issue:

```python
ValueError: '''You are passing a target array of shape (600, 1) while using as loss `categorical_crossentropy`.
`categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape (samples, classes).
If your targets are integer classes, you can convert them to the expected format via:

--
from keras.utils import to_categorical

y_binary = to_categorical(y_int)
--

Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.'''
```

Good error messages improve the productivity and the mood of your users.

<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr"><a href="https://twitter.com/fchollet?ref_src=twsrc%5Etfw">@fchollet</a> It&#39;s my first time using Keras and I think this is the nicest error message I&#39;ve ever received! It solved the problem too! <a href="https://t.co/N7n1tRBjrt">pic.twitter.com/N7n1tRBjrt</a></p>&mdash; Henry Dashwood (@hcndashwood) <a href="https://twitter.com/hcndashwood/status/928696266623737857?ref_src=twsrc%5Etfw">November 9, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


----

## Conclusion

These are all fairly simple principles, and following them will allow you to build APIs that people love to use. In turn, more people will start using your software, and you will achieve a greater impact in your field.

Always remember: software is for humans, not just for machines. Keep the user in mind at all times.

<br>
*[@fchollet](https://twitter.com/fchollet), November 2017*
<br>
----
