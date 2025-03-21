import pandas as pd
import random
import csv


# Function to generate sentiment dataset
def generate_sentiment_dataset(num_samples=500):
    """
    Generate a dataset of text, context, and sentiment labels for the assignment.
    Each sample contains:
    - Text: Short work/study related message (under 50 words)
    - Context: Supplementary information (under 20 words)
    - Label: Sentiment (Positive, Negative, or Neutral)
    """

    # Templates for generating varied content
    work_situations = [
        # Positive situations
        ("completed my project ahead of schedule", "The team was impressed with the results", "Positive"),
        ("received great feedback on my presentation today", "The client wants to expand our contract", "Positive"),
        ("got promoted after working hard all year", "My manager recognized my contributions", "Positive"),
        ("solved a difficult problem that was blocking the team", "Everyone was stuck on this for weeks", "Positive"),
        ("learned a new skill that will help with my career", "The training was very comprehensive", "Positive"),
        ("collaborated effectively with the marketing team", "We finished the campaign early", "Positive"),
        ("my code passed all tests on the first try", "The quality assurance team was pleased", "Positive"),
        ("finished all my assignments before the deadline", "I can now enjoy my weekend", "Positive"),
        ("our team exceeded the quarterly targets", "Management is planning a celebration", "Positive"),
        ("implemented a feature that users love", "Customer satisfaction scores increased", "Positive"),
        ("secured funding for our research project", "The competition was very tough", "Positive"),
        ("my article was accepted for publication", "The peer review process was rigorous", "Positive"),
        ("won an award for my innovative design", "The competition had many strong entries", "Positive"),
        ("received a bonus for exceptional work", "Company profits were higher than expected", "Positive"),
        ("found an elegant solution to a complex algorithm", "The optimization improved performance significantly",
         "Positive"),
        ("my mentee just got hired at a top company", "I've been coaching them for six months", "Positive"),
        ("fixed the bug that was causing system crashes", "Users had been complaining for weeks", "Positive"),
        ("successfully deployed to production with no issues", "The deployment process usually has problems",
         "Positive"),
        ("client approved our proposal without changes", "We put a lot of effort into the details", "Positive"),
        ("received recognition at the company all-hands", "The CEO personally thanked me", "Positive"),

        # Negative situations
        ("missed an important deadline today", "My manager is very disappointed", "Negative"),
        ("the client rejected our proposal", "We spent weeks preparing it", "Negative"),
        ("got negative feedback on my quarterly review", "I thought I was performing well", "Negative"),
        ("lost the data I was working on all day", "I forgot to save my progress", "Negative"),
        ("our team project is falling behind schedule", "We have limited resources to catch up", "Negative"),
        ("made a calculation error in the financial report", "The CFO noticed during the presentation", "Negative"),
        ("broke the build with my last commit", "The team can't continue their work", "Negative"),
        ("received a warning for being late again", "Traffic was terrible this morning", "Negative"),
        ("our funding request was denied", "We needed those resources for the project", "Negative"),
        ("failed the certification exam by two points", "I studied for months", "Negative"),
        ("lost a major client to our competitor", "They offered a lower price", "Negative"),
        ("my leave request was rejected", "I needed time off for personal reasons", "Negative"),
        ("the prototype failed during the demo", "The executives were all watching", "Negative"),
        ("received a poor performance evaluation", "I might not get the promotion", "Negative"),
        ("the server crashed during peak hours", "Users are complaining on social media", "Negative"),
        ("discovered a security vulnerability in our system", "It may have been exploited already", "Negative"),
        ("our team's budget was cut by 20 percent", "We'll have to scale back our plans", "Negative"),
        ("received complaints about my customer service", "My supervisor wants to meet tomorrow", "Negative"),
        ("my research hypothesis was disproven", "I spent a year collecting this data", "Negative"),
        ("accidentally sent confidential information to the wrong person", "I immediately reported the incident",
         "Negative"),

        # Neutral situations
        ("working on documentation for the new system", "It needs to be completed by next week", "Neutral"),
        ("attended a training session on data security", "The company is updating protocols", "Neutral"),
        ("moved to a different desk in the office", "The layout is being reorganized", "Neutral"),
        ("switched to a new project management tool", "The transition will take some time", "Neutral"),
        ("participating in the annual company survey", "Management will review the feedback", "Neutral"),
        ("updating my resume with recent experience", "It's good to keep it current", "Neutral"),
        ("scheduling meetings with stakeholders", "We need to align on project goals", "Neutral"),
        ("reading reports from the research team", "They contain useful information", "Neutral"),
        ("setting up my development environment", "I'm working on a new laptop", "Neutral"),
        ("organizing files before the system migration", "The IT team requested this", "Neutral"),
        ("taking inventory of office supplies", "It's part of monthly procedures", "Neutral"),
        ("waiting for approval to proceed with the project", "The decision committee meets tomorrow", "Neutral"),
        ("processing customer orders from yesterday", "The volume is about average", "Neutral"),
        ("reviewing changes to company policy", "HR sent an email about updates", "Neutral"),
        ("preparing for tomorrow's team meeting", "I need to summarize our progress", "Neutral"),
        ("backing up important files to the cloud", "It's a routine security measure", "Neutral"),
        ("compiling statistics for the monthly report", "The numbers look typical", "Neutral"),
        ("calibrating equipment for the next experiment", "Standard procedure before testing", "Neutral"),
        ("setting up video conference equipment", "The international team is joining remotely", "Neutral"),
        ("logging customer interactions in the system", "It's required for all support calls", "Neutral"),
    ]

    # Study situations
    study_situations = [
        # Positive study situations
        ("got an A on my final exam", "I studied consistently throughout the semester", "Positive"),
        ("understood a difficult concept in calculus", "The professor's explanation was very clear", "Positive"),
        ("completed my thesis draft ahead of schedule", "My advisor provided helpful feedback", "Positive"),
        ("my research paper was accepted by the journal", "The reviewers had minimal revisions", "Positive"),
        ("won a scholarship for academic excellence", "The competition was very strong this year", "Positive"),
        ("my study group solved all the practice problems", "We worked well together as a team", "Positive"),
        ("received praise for my lab report", "The experiment yielded interesting results", "Positive"),
        ("passed my comprehensive exams with distinction", "Years of preparation paid off", "Positive"),
        ("my presentation received the highest mark in class", "I practiced it many times", "Positive"),
        ("got accepted into my dream graduate program", "The admissions process was rigorous", "Positive"),
        ("successfully defended my dissertation", "The committee asked challenging questions", "Positive"),
        ("my professor asked to use my essay as an example", "It had a unique analytical approach", "Positive"),
        ("finished all my assignments for the week", "Now I can focus on my research project", "Positive"),
        ("collaborated successfully on a group project", "Everyone contributed equally", "Positive"),
        ("my experiment validated my hypothesis", "The results were statistically significant", "Positive"),
        ("received a teaching assistantship for next semester", "It will help with tuition costs", "Positive"),
        ("my academic paper won an award", "It represented months of research", "Positive"),
        ("improved my GPA this semester", "My new study habits are working well", "Positive"),
        ("mastered a programming language that will help my research", "The online course was excellent", "Positive"),
        (
        "received positive student evaluations for my teaching", "This was my first time as an instructor", "Positive"),

        # Negative study situations
        ("failed my midterm exam despite studying hard", "The questions were unexpectedly difficult", "Negative"),
        ("missed an important assignment deadline", "I misread the syllabus information", "Negative"),
        ("lost all my research data due to computer crash", "I didn't have a recent backup", "Negative"),
        ("got rejected from the internship I wanted", "They said I lack experience", "Negative"),
        ("my thesis proposal was rejected", "The committee had major concerns", "Negative"),
        ("struggling to understand key concepts in this course", "The textbook is confusing", "Negative"),
        ("my experiment yielded null results", "I spent months setting it up", "Negative"),
        ("received harsh criticism on my term paper", "The professor said my analysis was superficial", "Negative"),
        ("failed to get the research grant", "Funding is very competitive this year", "Negative"),
        ("my study group is not working well together", "There are conflicts about workload distribution", "Negative"),
        ("missed an important lecture due to illness", "The material will be on the exam", "Negative"),
        ("got a plagiarism warning for improper citations", "I didn't understand the citation format", "Negative"),
        ("my GPA dropped below the scholarship requirement", "I might lose my financial aid", "Negative"),
        ("professor rejected my topic for the final project", "I need to start over with a new idea", "Negative"),
        ("failed the laboratory safety exam", "I can't continue my experiments until I pass", "Negative"),
        ("lost my notes before the final exam", "I left my notebook on the bus", "Negative"),
        ("my advisor is going on sabbatical", "My research will be delayed", "Negative"),
        ("received conflicting feedback from different professors", "I don't know whose advice to follow", "Negative"),
        ("the required textbook is out of stock", "The assignment is due next week", "Negative"),
        ("my request for deadline extension was denied", "I explained my health situation", "Negative"),

        # Neutral study situations
        ("registered for next semester's classes", "The registration process was straightforward", "Neutral"),
        ("attended a workshop on research methods", "It was organized by the department", "Neutral"),
        ("ordered books for the upcoming term", "They should arrive next week", "Neutral"),
        ("meeting with my advisor to discuss progress", "It's our monthly check-in", "Neutral"),
        ("submitted my application for graduation", "The deadline is approaching", "Neutral"),
        ("taking notes during the biology lecture", "The topic is cellular respiration", "Neutral"),
        ("preparing for tomorrow's presentation", "It's worth 20% of the final grade", "Neutral"),
        ("studying in the library until closing time", "Exams start next Monday", "Neutral"),
        ("working on the literature review chapter", "I've compiled most of my sources", "Neutral"),
        ("transferring to a different research lab", "The focus aligns better with my interests", "Neutral"),
        ("attending office hours to clarify assignment requirements", "The instructions were ambiguous", "Neutral"),
        ("organizing my study schedule for finals week", "I have four exams in five days", "Neutral"),
        ("filling out forms for my study abroad program", "The application is due this Friday", "Neutral"),
        ("collecting data for my psychology experiment", "I need at least 50 more participants", "Neutral"),
        ("reviewing lecture slides before the quiz", "The material covers three weeks of content", "Neutral"),
        ("getting my student ID card replaced", "The old one was damaged", "Neutral"),
        ("updating my academic CV with recent publications", "I'm applying for a research position", "Neutral"),
        ("calibrating equipment for the chemistry lab", "Precision is essential for these experiments", "Neutral"),
        ("submitting my thesis to the review committee", "The defense is scheduled next month", "Neutral"),
        ("requesting transcripts for job applications", "Several positions require official records", "Neutral"),
    ]

    # Combine all situations
    all_situations = work_situations + study_situations

    # Ensure we have enough templates
    if num_samples > len(all_situations):
        # If we need more samples than templates, we'll generate variations
        additional_needed = num_samples - len(all_situations)

        # Create variations by mixing and matching texts and contexts
        for i in range(additional_needed):
            # Randomly select elements from different templates
            base1 = random.choice(all_situations)
            base2 = random.choice(all_situations)

            # Create new combination (keep sentiment aligned with the text)
            new_situation = (base1[0], base2[1], base1[2])
            all_situations.append(new_situation)

    # Shuffle and select the required number of samples
    random.shuffle(all_situations)
    selected_situations = all_situations[:num_samples]

    # Create the dataset
    data = []
    for text, context, label in selected_situations:
        data.append([text, context, label])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['text', 'context', 'label'])

    # Check balance of labels
    label_counts = df['label'].value_counts()
    print("Label distribution:")
    print(label_counts)

    return df


# Generate the dataset with 550 samples (extra for safety)
sentiment_df = generate_sentiment_dataset(550)

# Check for and remove any empty values
sentiment_df = sentiment_df.dropna()

# Ensure we have at least 500 samples after cleaning
if len(sentiment_df) < 500:
    print(f"Warning: After cleaning, only {len(sentiment_df)} samples remain. Generating more...")
    additional_df = generate_sentiment_dataset(600 - len(sentiment_df))
    additional_df = additional_df.dropna()
    sentiment_df = pd.concat([sentiment_df, additional_df]).reset_index(drop=True)
    sentiment_df = sentiment_df.iloc[:500]

# Final verification
print(f"Final dataset size: {len(sentiment_df)}")

# Save to CSV
sentiment_df.to_csv('sentiment_data.csv', index=False, quoting=csv.QUOTE_ALL)
print("Dataset saved to 'sentiment_data.csv'")