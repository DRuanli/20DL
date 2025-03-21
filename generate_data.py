import pandas as pd
import numpy as np
import random
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define sentiment classes and their associated keywords/patterns
SENTIMENT_PATTERNS = {
    'Positive': [
        'accomplished', 'achieved', 'amazing', 'awesome', 'delighted',
        'excellent', 'excited', 'fantastic', 'glad', 'great', 'happy',
        'impressed', 'love', 'outstanding', 'perfect', 'pleased',
        'proud', 'succeeded', 'thank', 'thrilled', 'wonderful'
    ],
    'Negative': [
        'annoyed', 'awful', 'bad', 'disappointed', 'disaster',
        'fail', 'frustrated', 'hate', 'horrible', 'late', 'missed',
        'poor', 'sad', 'sorry', 'stressed', 'struggled', 'terrible',
        'tired', 'unhappy', 'upset', 'worried'
    ],
    'Neutral': [
        'acceptable', 'average', 'common', 'completed', 'fine',
        'normal', 'okay', 'regular', 'standard', 'typical',
        'usual', 'submitted', 'finished', 'attended', 'working'
    ]
}

# Work/study related contexts
CONTEXTS = {
    'Positive': [
        "The team was supportive.",
        "Manager offered encouragement.",
        "Deadline was extended.",
        "Received extra help.",
        "Got positive feedback.",
        "Team celebrated success.",
        "Resources were provided.",
        "Given a flexible schedule.",
        "Offered a promotion.",
        "Workload was manageable.",
        "Everyone contributed equally.",
        "The client was pleased.",
        "Project finished ahead of schedule.",
        "Meeting was productive.",
        "Teammates were responsive."
    ],
    'Negative': [
        "The deadline was tight.",
        "Manager was disappointed.",
        "Team morale was low.",
        "Project was understaffed.",
        "Client was unhappy.",
        "Budget was cut.",
        "Equipment malfunctioned.",
        "Meeting ran overtime.",
        "Communication broke down.",
        "Expectations were unclear.",
        "Insufficient training provided.",
        "Colleagues were unresponsive.",
        "Too many distractions present.",
        "Resources were limited.",
        "Technical issues occurred."
    ],
    'Neutral': [
        "Regular team meeting.",
        "Standard procedure followed.",
        "Normal working hours.",
        "Routine project update.",
        "Average client interaction.",
        "Standard workload assigned.",
        "Regular progress review.",
        "Usual team composition.",
        "Typical office environment.",
        "Standard project timeline.",
        "Regular weekly schedule.",
        "Standard industry practices.",
        "Normal office protocol.",
        "Followed established guidelines.",
        "Expected project parameters."
    ]
}

# Work/study related texts
TEXTS = {
    'Positive': [
        "I completed the project ahead of schedule.",
        "My presentation went really well today.",
        "I received great feedback on my report.",
        "Our team exceeded the quarterly targets.",
        "The client approved our proposal immediately.",
        "I solved a difficult problem elegantly.",
        "My code runs faster after optimization.",
        "I passed all my exams with high grades.",
        "My research paper was accepted for publication.",
        "I learned a valuable new skill today.",
        "The collaboration with other departments was excellent.",
        "The professor praised my thesis draft.",
        "Our prototype received unanimous approval.",
        "I successfully defended my dissertation.",
        "My experiment yielded significant results.",
        "The team appreciated my contributions.",
        "I found a creative solution to our challenge.",
        "My work-life balance has improved greatly.",
        "I received recognition for my efforts.",
        "Our study confirmed our hypothesis."
    ],
    'Negative': [
        "I missed an important deadline today.",
        "My presentation had technical issues.",
        "The client rejected our proposal.",
        "I made a critical error in my calculations.",
        "Our team failed to meet the target.",
        "I'm overwhelmed with the workload.",
        "My code is full of bugs I can't fix.",
        "I failed the midterm exam.",
        "My application for funding was rejected.",
        "I lost all my work due to a system crash.",
        "The meeting was a complete waste of time.",
        "My manager criticized my report harshly.",
        "I couldn't answer the professor's questions.",
        "The project is significantly over budget.",
        "My research hit a major obstacle.",
        "I missed important information in the lecture.",
        "The client was dissatisfied with our progress.",
        "I'm falling behind on my coursework.",
        "My experiment produced contradictory results.",
        "The team dynamics are creating conflicts."
    ],
    'Neutral': [
        "I submitted my report on time.",
        "The meeting lasted one hour as scheduled.",
        "I attended the training session.",
        "I'm working on the assigned project.",
        "The client requested some revisions.",
        "I documented the process as required.",
        "The code review is scheduled for tomorrow.",
        "I'm preparing for the upcoming exam.",
        "The team discussed the project timeline.",
        "I received the task assignment for next week.",
        "I took notes during the lecture.",
        "We implemented the standard protocol.",
        "The office closed at the regular time.",
        "I filled out the required paperwork.",
        "I participated in the department meeting.",
        "The course follows the usual curriculum.",
        "I use the company-provided equipment.",
        "We follow the established workflow.",
        "I maintain regular office hours.",
        "The project uses standard methodology."
    ]
}


def generate_text_variations(base_texts, sentiment, count=50):
    """Generate variations of base texts to expand the dataset"""
    variations = []
    sentiment_words = SENTIMENT_PATTERNS[sentiment]

    for _ in range(count):
        # Select a random base text
        text = random.choice(base_texts)

        # Apply random transformations
        r = random.random()

        if r < 0.25 and len(word_tokenize(text)) < 45:  # Add sentiment words
            words = text.split()
            insert_pos = random.randint(1, len(words) - 1)
            sentiment_word = random.choice(sentiment_words)

            if text.endswith("."):
                words.insert(insert_pos, sentiment_word)
                text = " ".join(words)
            else:
                words.append("and it was " + sentiment_word)
                text = " ".join(words)

        elif r < 0.5:  # Change beginning
            if text.startswith("I "):
                prefix = random.choice(["Today I ", "This morning I ", "This week I ", "Recently I "])
                text = prefix + text[2:]
            elif text.startswith("My "):
                text = random.choice(["The ", "Our "]) + text[3:]
            elif text.startswith("Our "):
                text = random.choice(["The ", "My "]) + text[4:]

        elif r < 0.75 and not text.endswith("."):  # Add ending
            text += random.choice([
                " as expected.",
                " surprisingly.",
                " for the first time.",
                " again.",
                " finally."
            ])

        # Ensure text is not too long (under 50 words)
        if len(word_tokenize(text)) <= 50:
            variations.append(text)
        else:
            # If too long, just use a base text
            variations.append(random.choice(base_texts))

    return variations


def generate_context_variations(base_contexts, count=50):
    """Generate variations of contexts"""
    variations = []

    for _ in range(count):
        context = random.choice(base_contexts)

        # Apply minor transformations to contexts
        r = random.random()

        if r < 0.3 and len(word_tokenize(context)) < 18:  # Add qualifier
            if context.endswith("."):
                context = context[:-1] + random.choice([
                    " this time.",
                    " as usual.",
                    " unexpectedly.",
                    " today."
                ])
            else:
                context += random.choice([
                    " this time",
                    " as usual",
                    " unexpectedly",
                    " today"
                ])

        elif r < 0.6:  # Change beginning for variety
            if context.startswith("The "):
                context = random.choice(["Our ", "Their ", "A "]) + context[4:]

        # Ensure context is not too long (under 20 words)
        if len(word_tokenize(context)) <= 20:
            variations.append(context)
        else:
            # If too long, use base context
            variations.append(random.choice(base_contexts))

    return variations


def generate_sentiment_dataset(min_samples=500):
    """Generate the full sentiment dataset with balanced classes"""
    data = []
    samples_per_class = min_samples // 3 + 10  # Add a buffer

    for sentiment in ['Positive', 'Negative', 'Neutral']:
        # Generate variations of texts and contexts
        texts = generate_text_variations(TEXTS[sentiment], sentiment, samples_per_class)
        contexts = generate_context_variations(CONTEXTS[sentiment], samples_per_class)

        # Create samples with matching sentiment
        for i in range(samples_per_class):
            text = texts[i % len(texts)]
            context = contexts[i % len(contexts)]

            # Ensure text and context are within word limits
            text_words = len(word_tokenize(text))
            context_words = len(word_tokenize(context))

            if text_words <= 50 and context_words <= 20:
                data.append({
                    'text': text,
                    'context': context,
                    'label': sentiment
                })

    # Shuffle the dataset
    random.shuffle(data)

    return data


def main():
    # Generate dataset
    print("Generating sentiment dataset...")
    dataset = generate_sentiment_dataset(550)  # Generate extra to account for potential filtering

    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    # Check for empty rows
    df = df.dropna()
    print(f"After removing empty rows: {len(df)} samples")

    # Verify we have at least 500 samples
    if len(df) < 500:
        print("Warning: Dataset has fewer than 500 samples after cleaning.")
        print("Generating additional samples...")
        more_data = generate_sentiment_dataset(550 - len(df))
        df = pd.concat([df, pd.DataFrame(more_data)], ignore_index=True)
        df = df.dropna()

    # Verify class balance
    class_counts = df['label'].value_counts()
    print("\nClass distribution:")
    print(class_counts)

    # Check text and context length
    df['text_length'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    df['context_length'] = df['context'].apply(lambda x: len(word_tokenize(x)))

    print(
        f"\nText length stats: Min={df['text_length'].min()}, Max={df['text_length'].max()}, Avg={df['text_length'].mean():.1f}")
    print(
        f"Context length stats: Min={df['context_length'].min()}, Max={df['context_length'].max()}, Avg={df['context_length'].mean():.1f}")

    # Remove length columns for final dataset
    df = df.drop(['text_length', 'context_length'], axis=1)

    # Take exactly 500 samples if we have more
    if len(df) > 500:
        # Ensure balanced classes
        final_df = pd.DataFrame()
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            sentiment_df = df[df['label'] == sentiment]
            samples_per_class = min(len(sentiment_df), 500 // 3)
            final_df = pd.concat([final_df, sentiment_df.sample(samples_per_class)], ignore_index=True)

        # If we don't have exactly 500 samples, add some more randomly
        remaining = 500 - len(final_df)
        if remaining > 0:
            excluded = df[~df.index.isin(final_df.index)]
            final_df = pd.concat([final_df, excluded.sample(remaining)], ignore_index=True)

        df = final_df

    # Save to CSV
    df.to_csv('sentiment_data.csv', index=False)
    print(f"\nSaved {len(df)} samples to sentiment_data.csv")

    # Show some examples
    print("\nExample data:")
    for i, example in enumerate(df.sample(3).itertuples()):
        print(f"\nExample {i + 1}:")
        print(f"Text: \"{example.text}\"")
        print(f"Context: \"{example.context}\"")
        print(f"Label: {example.label}")


if __name__ == "__main__":
    main()