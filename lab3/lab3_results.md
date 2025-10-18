# Names: Joseph Nguyen and Riley Smith
# Lab: Lab3 (Advanced Prompt Engineering)
# Date: October 17, 2025

1. **Task 3.2**; The Wolf, Goat, and Cabbage
    - There is no difference in the way the LLM solved the logic puzzle compared to Wikipedia's answer. The LLM provided an efficient response.
2. **Task 6.1.2**
    - In the first snake game iteration, the game worked as expected. When the snake ate a dot, it grew by one unit. When the snake travelled out of bounds, or ran into itself, the game was over. Also, I was not allowed to go back in the direction that I was currently travelling.
    - In the second snake game iteration, with the inclusion of 2 new features to the game, the code had an UnboundLocalError. This is when local variable is being referenced before it has been assigned a value. After I asked the LLM to revise the code given the error message, the resulting game was playable. However, it did not include any new features, the snake did not grew larger than 1 unit after eating dots, and could travel in the opposite direction that it was currently travelling. The game still ended when the snake went out of bounds. The LLM seemed to not have great conversation history.