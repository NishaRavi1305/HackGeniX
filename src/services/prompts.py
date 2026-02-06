"""
Interview Question Prompts and Templates.

Contains structured prompts for generating interview questions,
evaluating answers, and conducting different interview stages.
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class InterviewStage(str, Enum):
    """Interview stages."""
    SCREENING = "screening"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SYSTEM_DESIGN = "system_design"
    WRAP_UP = "wrap_up"


class QuestionDifficulty(str, Enum):
    """Question difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class QuestionTemplate:
    """Template for generating questions."""
    stage: InterviewStage
    difficulty: QuestionDifficulty
    prompt: str
    follow_up_prompt: str
    expected_duration_seconds: int = 120


# System prompts for the AI interviewer
INTERVIEWER_SYSTEM_PROMPT = """You are an expert technical interviewer for a software engineering position. 
Your role is to:
1. Generate relevant, insightful interview questions based on the job description and candidate's resume
2. Assess candidate responses fairly and objectively
3. Ask appropriate follow-up questions to probe deeper understanding
4. Maintain a professional but friendly tone throughout

Key guidelines:
- Questions should be specific to the role requirements
- Tailor difficulty based on the candidate's experience level
- Focus on practical, real-world scenarios
- Avoid trick questions or gotchas
- Be encouraging while maintaining rigor"""


# Question generation prompts
QUESTION_GENERATION_PROMPTS = {
    InterviewStage.SCREENING: """Generate {num_questions} screening questions for a {role_title} position.

Job Description Summary:
{jd_summary}

Candidate Background:
{resume_summary}

Requirements:
- Questions should verify basic qualifications
- Include at least one question about career motivation
- Keep questions concise (answerable in 1-2 minutes)
- Difficulty: Easy to Medium

Output format (JSON array):
[
  {{
    "question": "The question text",
    "purpose": "What this question assesses",
    "expected_answer_points": ["key point 1", "key point 2"],
    "difficulty": "easy|medium",
    "duration_seconds": 60
  }}
]""",

    InterviewStage.TECHNICAL: """Generate {num_questions} technical interview questions for a {role_title} position.

Required Skills:
{required_skills}

Candidate's Technical Background:
{technical_background}

Focus Areas:
{focus_areas}

Requirements:
- Questions should test practical knowledge, not memorization
- Include coding/problem-solving scenarios where appropriate
- Mix of conceptual and hands-on questions
- Progressive difficulty (start medium, can go to hard)

Output format (JSON array):
[
  {{
    "question": "The question text",
    "category": "algorithms|system-design|language-specific|domain-knowledge",
    "purpose": "What this question assesses",
    "expected_answer_points": ["key point 1", "key point 2", "key point 3"],
    "follow_up_questions": ["follow-up 1", "follow-up 2"],
    "difficulty": "medium|hard",
    "duration_seconds": 180
  }}
]""",

    InterviewStage.BEHAVIORAL: """Generate {num_questions} behavioral interview questions for a {role_title} position.

Job Requirements:
{jd_summary}

Candidate Experience:
{experience_summary}

Key Competencies to Assess:
{competencies}

Requirements:
- Use STAR format prompts (Situation, Task, Action, Result)
- Focus on past experiences that predict future performance
- Include questions about teamwork, challenges, and growth
- Assess cultural fit and soft skills

Output format (JSON array):
[
  {{
    "question": "The question text (STAR format)",
    "competency": "leadership|teamwork|problem-solving|communication|adaptability",
    "purpose": "What this question assesses",
    "red_flags": ["warning sign 1", "warning sign 2"],
    "green_flags": ["positive indicator 1", "positive indicator 2"],
    "difficulty": "medium",
    "duration_seconds": 180
  }}
]""",

    InterviewStage.SYSTEM_DESIGN: """Generate {num_questions} system design questions for a {role_title} position.

Technical Requirements:
{technical_requirements}

Candidate's System Experience:
{system_experience}

Scale/Complexity Level:
{complexity_level}

Requirements:
- Questions should be open-ended design problems
- Include scalability considerations
- Appropriate for candidate's experience level
- Focus on practical, real-world systems

Output format (JSON array):
[
  {{
    "question": "Design a system that...",
    "category": "distributed-systems|data-intensive|real-time|microservices",
    "key_components": ["component 1", "component 2"],
    "expected_discussion_points": ["scalability", "reliability", "trade-offs"],
    "follow_up_questions": ["What if we need to handle 10x traffic?"],
    "difficulty": "hard",
    "duration_seconds": 300
  }}
]""",
}


# Answer evaluation prompts
ANSWER_EVALUATION_PROMPT = """Evaluate the candidate's answer to an interview question.

Question: {question}

Expected Answer Points:
{expected_points}

Candidate's Answer:
{answer}

Evaluate on these criteria:
1. Technical Accuracy (0-100): Is the answer factually correct?
2. Completeness (0-100): Did they address all key points?
3. Clarity (0-100): Was the explanation clear and well-structured?
4. Depth (0-100): Did they demonstrate deep understanding?

Also provide:
- Key strengths in the answer
- Areas for improvement
- Suggested follow-up question (if needed)
- Overall recommendation

Output format (JSON):
{{
  "scores": {{
    "technical_accuracy": 0-100,
    "completeness": 0-100,
    "clarity": 0-100,
    "depth": 0-100,
    "overall": 0-100
  }},
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["area 1", "area 2"],
  "follow_up_question": "optional follow-up or null",
  "recommendation": "strong|acceptable|weak|insufficient",
  "notes": "Brief evaluation summary"
}}"""


BEHAVIORAL_EVALUATION_PROMPT = """Evaluate the candidate's behavioral answer using STAR criteria.

Question: {question}
Competency Being Assessed: {competency}

Candidate's Answer:
{answer}

STAR Evaluation:
1. Situation (0-25): Did they clearly describe the context?
2. Task (0-25): Did they explain their specific responsibility?
3. Action (0-25): Did they describe concrete actions they took?
4. Result (0-25): Did they share measurable outcomes?

Also assess:
- Authenticity: Does this seem like a genuine experience?
- Relevance: Is this example relevant to the role?
- Self-awareness: Did they show reflection and learning?

Red flags to watch for:
{red_flags}

Green flags (positive indicators):
{green_flags}

Output format (JSON):
{{
  "star_scores": {{
    "situation": 0-25,
    "task": 0-25,
    "action": 0-25,
    "result": 0-25,
    "total": 0-100
  }},
  "authenticity_score": 0-100,
  "relevance_score": 0-100,
  "self_awareness_score": 0-100,
  "overall_score": 0-100,
  "red_flags_detected": ["flag if any"],
  "green_flags_detected": ["flag if any"],
  "recommendation": "strong|acceptable|weak|concerning",
  "notes": "Brief evaluation summary"
}}"""


# Follow-up question generation
FOLLOW_UP_PROMPT = """Based on the candidate's answer, generate an appropriate follow-up question.

Original Question: {original_question}

Candidate's Answer:
{answer}

Evaluation Summary:
{evaluation_summary}

Guidelines:
- If answer was incomplete, probe for missing details
- If answer was vague, ask for specific examples
- If answer showed expertise, dig deeper into advanced topics
- If answer revealed potential weakness, explore tactfully

Generate ONE follow-up question that will best assess the candidate's true capabilities.

Output format (JSON):
{{
  "follow_up_question": "The question text",
  "purpose": "Why this follow-up is valuable",
  "expected_depth": "What a good answer would include"
}}"""


# Adaptive difficulty prompts
DIFFICULTY_ADJUSTMENT_PROMPT = """Based on the candidate's performance so far, determine the appropriate difficulty for the next question.

Performance Summary:
- Questions answered: {questions_answered}
- Average score: {average_score}
- Trend: {trend}  (improving/declining/stable)
- Strongest areas: {strong_areas}
- Weakest areas: {weak_areas}

Current difficulty level: {current_difficulty}

Determine:
1. Should difficulty increase, decrease, or stay the same?
2. Which topic areas need more probing?
3. Any areas to skip (already demonstrated mastery)?

Output format (JSON):
{{
  "next_difficulty": "easy|medium|hard",
  "difficulty_change": "increase|decrease|maintain",
  "reason": "Brief explanation",
  "focus_areas": ["area to probe more"],
  "skip_areas": ["area already demonstrated"],
  "candidate_trajectory": "strong|on-track|struggling"
}}"""


# Report generation prompts
INTERVIEW_SUMMARY_PROMPT = """Generate a comprehensive interview summary and hiring recommendation.

Position: {role_title}
Candidate: {candidate_name}
Interview Duration: {duration_minutes} minutes

Performance by Stage:
{stage_performances}

Question-by-Question Breakdown:
{question_breakdown}

Scoring Summary:
- Technical Skills: {technical_score}/100
- Behavioral/Soft Skills: {behavioral_score}/100
- Communication: {communication_score}/100
- Problem Solving: {problem_solving_score}/100
- Cultural Fit: {cultural_fit_score}/100
- Overall: {overall_score}/100

Generate:
1. Executive Summary (2-3 sentences)
2. Key Strengths (top 3-5)
3. Areas of Concern (if any)
4. Hiring Recommendation with confidence level
5. Suggested next steps

Output format (JSON):
{{
  "executive_summary": "Brief overall assessment",
  "strengths": [
    {{"area": "strength area", "evidence": "specific example from interview"}}
  ],
  "concerns": [
    {{"area": "concern area", "severity": "minor|moderate|major", "evidence": "specific example"}}
  ],
  "recommendation": "strong_hire|hire|no_hire|strong_no_hire",
  "confidence": 0-100,
  "reasoning": "Detailed justification for recommendation",
  "next_steps": ["suggested action 1", "suggested action 2"],
  "comparison_notes": "How candidate compares to typical candidates for this role"
}}"""


# Hallucination detection prompt
HALLUCINATION_CHECK_PROMPT = """Verify that the evaluation is grounded in the candidate's actual response.

Original Question: {question}

Candidate's Answer:
{answer}

Generated Evaluation:
{evaluation}

Check for:
1. Does the evaluation only reference things actually said by the candidate?
2. Are the scores justified by specific parts of the answer?
3. Are there any assumptions or inferences not supported by the text?

Output format (JSON):
{{
  "is_grounded": true|false,
  "issues": [
    {{"type": "unsupported_claim|score_not_justified|missing_evidence", "description": "..."}}
  ],
  "confidence": 0-100,
  "corrected_evaluation": null or corrected JSON if issues found
}}"""
