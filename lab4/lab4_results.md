# Names: Joseph Nguyen and Joshua Bielas
# Lab: Lab4 (Retrieval Augmented Generation)
# Date: 10/27/25

### 5.1.1; Does the structure of the questions seem to matter?
* Yes; Phrasing a question in an open-ended way provides an answer if it's in the knowledge base. But, if we ask a similar question in close-ended way, it does not provide an answer. For example, the question "What is the CS450 Week of Worship" did not give an answer but the question " Tell me about the Week of Worship" did.

### 5.1.2; Would you say that RAG has limitations based on this exercise?
* Yes it has limitations; How the question is phrased (closed-ended or open-ended) can determine if the LLM using RAG provides an answer.

### 5.1.3; Did you notice anything about the course name in the questions? Is that course name actually in the context? What does this demonstrate?
* The questions refer to the course as "CS450", whereas the course name in the syllabus "CPTR 450". There is no mention of "CS450" in the syllabus. The questions being succesfully answered demonstrates that from the information it was trained on, it can determine that "CPTR 450" and "CS450" are interchangeable.