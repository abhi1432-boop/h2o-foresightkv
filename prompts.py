"""Diverse prompt set for collecting attention traces.

Covers reasoning, factual recall, conversational, code, creative writing, and
multi-step instruction following. Each prompt is in Phi-2's "Instruct/Output"
format so the model engages in genuine generation, not echo-completion.

Split into TRAIN_PROMPTS and EVAL_PROMPTS. Eval set is held out from labelling
so the trained scorer is judged on prompts it never saw.
"""

_QA = [
    "What is the capital of France?",
    "Who wrote the play Hamlet?",
    "What is the boiling point of water in Celsius?",
    "Name the planets of the solar system in order from the sun.",
    "What language is spoken in Brazil?",
    "Who painted the Mona Lisa?",
    "What is the largest mammal on Earth?",
    "Which element has the chemical symbol Au?",
    "What year did World War II end?",
    "Who developed the theory of general relativity?",
    "What is the tallest mountain in the world?",
    "Name three primary colors.",
    "What is the speed of light in vacuum?",
    "Which organ pumps blood through the human body?",
    "What is the currency of Japan?",
    "Who is credited with inventing the telephone?",
    "What is the smallest prime number?",
    "Name the seven continents.",
    "What gas do plants absorb during photosynthesis?",
    "In which country is the city of Cairo located?",
]

_REASONING = [
    "If a train leaves at 3pm going 60 mph and another leaves at 4pm going 80 mph, when does the second catch up?",
    "A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?",
    "If five machines make five widgets in five minutes, how long does it take 100 machines to make 100 widgets?",
    "A farmer has 17 sheep, all but 9 run away. How many are left?",
    "I have three apples today. Yesterday I ate one apple. How many apples do I have now?",
    "If you are running a race and pass the person in second place, what place are you in?",
    "A doctor gives you three pills, telling you to take one every half hour. How long do the pills last?",
    "Mary's father has five daughters: Nana, Nene, Nini, Nono. What is the fifth daughter's name?",
    "Some months have 31 days, some have 30. How many have 28?",
    "If it takes 8 men 10 hours to build a wall, how long for 4 men working at the same rate?",
    "A man builds a house with all four walls facing south. A bear walks by. What color is the bear?",
    "If two's company and three's a crowd, what are four and five?",
    "How many times can you subtract 10 from 100?",
    "What's heavier, a pound of feathers or a pound of bricks?",
    "If you have a 5-gallon and a 3-gallon jug, how do you measure exactly 4 gallons?",
    "I am thinking of a number. Double it, add 10, divide by 2, subtract the original number. What is the result?",
    "If yesterday was tomorrow, today would be Sunday. What day is it actually?",
    "A clock shows 3:15. What is the angle between the hour and minute hands?",
    "Three friends split a $30 bill, paying $10 each. The waiter returns $5, keeps $2, gives back $1 to each. Where is the missing dollar?",
    "If a hen and a half lay an egg and a half in a day and a half, how many eggs do six hens lay in six days?",
]

_CONVERSATIONAL = [
    "Tell me about your favorite season and why you like it.",
    "Describe what makes a good friend in a few sentences.",
    "What advice would you give to someone starting a new job?",
    "How would you explain the concept of patience to a child?",
    "What's a good way to relax after a stressful day?",
    "Share a tip for staying focused while studying.",
    "Describe a perfect Saturday morning.",
    "How do you politely decline an invitation?",
    "What makes a meal memorable?",
    "Explain why exercise is important.",
    "How would you welcome a new neighbor?",
    "Suggest a fun activity for a rainy weekend.",
    "What does kindness look like in everyday life?",
    "How would you comfort a friend who lost a pet?",
    "Describe the feeling of finishing a long book.",
    "What is one habit you wish more people had?",
    "How do you handle disagreements with friends?",
    "What's the best way to celebrate a birthday?",
    "Share a tip for cooking a simple but tasty dinner.",
    "How would you describe the sound of rain to someone who has never heard it?",
]

_CODE = [
    "Write a Python function that reverses a string.",
    "Show me a Python loop that prints the numbers from 1 to 10.",
    "Write a Python function that checks if a number is prime.",
    "Give me a Python snippet to read a text file line by line.",
    "Write a Python function that returns the factorial of n.",
    "Show a Python list comprehension that squares even numbers from 0 to 20.",
    "Write a function that returns the Fibonacci sequence up to n terms.",
    "Show me how to sort a dictionary by value in Python.",
    "Write a function that counts vowels in a string.",
    "Demonstrate try/except in Python with a divide by zero example.",
    "Write Python code to find duplicates in a list.",
    "Show how to use a dictionary comprehension in Python.",
    "Write a function that flattens a nested list.",
    "Demonstrate Python f-strings with a name and age example.",
    "Write a function that returns the longest word in a sentence.",
    "Show me how to open a JSON file in Python.",
    "Write a function that converts Celsius to Fahrenheit.",
    "Show how to use map and filter in Python.",
    "Write a Python function that checks if a string is a palindrome.",
    "Show me how to define a class with an init method in Python.",
]

_CREATIVE = [
    "Write a haiku about a cat.",
    "Compose a short poem about autumn leaves.",
    "Tell a two-sentence horror story.",
    "Write the opening line of a mystery novel.",
    "Describe a sunset in three sentences.",
    "Write a tiny fable about a clever fox.",
    "Imagine a city built on clouds. Describe it briefly.",
    "Write a poem about the moon in four lines.",
    "Tell a short story about a robot learning to laugh.",
    "Write a haiku about coffee.",
    "Describe a dragon that loves gardening.",
    "Write a limerick about a programmer.",
    "Compose a short prayer for travelers.",
    "Tell a tiny story set entirely inside a library.",
    "Write a description of a haunted lighthouse.",
    "Imagine a song a tree might hum. Write the first verse.",
    "Describe what music tastes like.",
    "Write a short tale about a lonely cloud.",
    "Compose a poem about an old wooden door.",
    "Tell a story about a key that fits no lock.",
]

_FACTUAL_LONG = [
    "Explain how photosynthesis works at a basic level.",
    "Describe the main causes of the French Revolution.",
    "Summarize what plate tectonics means.",
    "Explain how vaccines train the immune system.",
    "Describe what a black hole is and how it forms.",
    "Explain the basic structure of an atom.",
    "Summarize the role of DNA in living cells.",
    "Describe how the water cycle works.",
    "Explain why the sky appears blue.",
    "Describe the difference between weather and climate.",
    "Explain how a refrigerator keeps food cold.",
    "Summarize the contribution of Marie Curie to science.",
    "Describe how bees make honey.",
    "Explain how the human eye perceives color.",
    "Describe the function of the liver in the body.",
    "Explain what an ecosystem is using a forest example.",
    "Describe how a volcano forms and erupts.",
    "Explain the basic idea of natural selection.",
    "Describe how an electric motor works.",
    "Summarize why the seasons happen.",
]

_INSTRUCTIONS = [
    "List five tips for writing clear emails.",
    "Give me a step by step guide to brewing tea.",
    "Outline the steps to change a flat tire.",
    "List five ways to save electricity at home.",
    "Provide a checklist for preparing a job interview.",
    "Outline the steps to plant a small herb garden.",
    "List five healthy breakfast ideas.",
    "Give a checklist for packing a weekend trip.",
    "Outline how to write a basic resume.",
    "List five tips for getting better sleep.",
    "Give step by step instructions to make a paper airplane.",
    "Outline a quick warm-up routine before exercise.",
    "List five ways to remember someone's name.",
    "Provide a checklist for buying a used car.",
    "Outline how to set up a simple budget.",
    "List five ways to be a better listener.",
    "Give step by step instructions for boiling an egg.",
    "Outline a five-minute morning stretching routine.",
    "List five tips for staying motivated.",
    "Give step by step instructions for ironing a shirt.",
]

_ALL = _QA + _REASONING + _CONVERSATIONAL + _CODE + _CREATIVE + _FACTUAL_LONG + _INSTRUCTIONS


def _format(prompts):
    return [f"Instruct: {p}\nOutput:" for p in prompts]


# 140 total; split 110 train, 30 eval. Held-out prompts never used for labelling.
ALL_PROMPTS = _format(_ALL)
TRAIN_PROMPTS = ALL_PROMPTS[:110]
EVAL_PROMPTS = ALL_PROMPTS[110:]


if __name__ == "__main__":
    print(f"total prompts: {len(ALL_PROMPTS)}")
    print(f"train: {len(TRAIN_PROMPTS)}  eval: {len(EVAL_PROMPTS)}")
    print("sample:", ALL_PROMPTS[0])
