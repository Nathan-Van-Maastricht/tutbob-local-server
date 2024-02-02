from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str
    remaining_seconds: int = 0  # Users should be aware of their remaining transcription time.


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    username: str
    password: str

class LectureBase(BaseModel):
    title: Optional[str] = None
    date_time: Optional[datetime] = None

class LectureCourse(BaseModel):
    id: Optional[int] = None
    title: Optional[str] = None
    date_time: Optional[datetime] = None

class CourseWithLectures(BaseModel):
    course_name: str
    course_id: int
    lectures: List[LectureCourse]

class LectureID(BaseModel):
    id: int

class LectureCreate(LectureBase):
    pass

class Lecture(LectureBase):
    id: int
    user_id: int
    class Config:
        orm_mode = True

class LectureUpdate(BaseModel):
    id : int
    title: str
    course_id: int

class User(UserBase):
    id: int
    # this will allow you to return a user's lectures when returning a User
    lectures: List[Lecture] = []
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str = None


class NoteBase(BaseModel):
    heading: Optional[str] = None
    notes: Optional[str] = None

class NoteCreate(NoteBase):
    lecture_id: int

class Note(NoteBase):
    id: int
    lecture_id: int

    class Config:
        orm_mode = True

class KeyConceptBase(BaseModel):
    concept_text: Optional[str] = None
    explanation_text: Optional[str] = None
    simplify_text: Optional[str] = None
    elaborate_text: Optional[str] = None

class KeyConceptReviewBase(BaseModel):
    concept_text: str
    explanation_text: str

class KeyConceptCreate(KeyConceptBase):
    pass

class KeyConcept(KeyConceptBase):
    id: int
    review_id: int

    class Config:
        orm_mode = True

class VectorEmbeddingBase(BaseModel):
    lecture_id: int
    transcript: str

class VectorEmbeddingCreate(VectorEmbeddingBase):
    vector: str

class VectorEmbeddingRead(VectorEmbeddingBase):
    id: int
    lecture_id: int
    vector: str

    class Config:
        orm_mode = True

class RealTimeTranscriptBase(BaseModel):
    lecture_id: int
    transcript_text: str
    timestamp: Optional[datetime] = None

class RealTimeTranscriptCreate(RealTimeTranscriptBase):
    pass

class RealTimeTranscript(RealTimeTranscriptBase):
    id: int

    class Config:
        orm_mode = True

class ReviewBase(BaseModel):
    interval: str
    summary: str

class ReviewCreate(ReviewBase):
    key_concepts: List[KeyConceptCreate] = []
    lecture_id: int

class Review(ReviewBase):
    id: int
    lecture_id: int

    class Config:
        orm_mode = True


class AnswerBase(BaseModel):
    answer_text: str
    answer_option: str

class AnswerCreate(AnswerBase):
    pass

class Answer(AnswerBase):
    id: int
    question_id: int

    class Config:
        orm_mode = True

class QuestionBase(BaseModel):
    question: str
    explanation: str
    correct_answer: str

class QuestionCreate(QuestionBase):
    answers: List[AnswerCreate]

class Question(QuestionBase):
    id: int
    quiz_id: Optional[int]  # This is optional because you might not always have this information
    answers: List[Answer]

    class Config:
        orm_mode = True

class QuizBase(BaseModel):
    lecture_id: Optional[int]

class QuizCreate(QuizBase):
    questions: List[QuestionCreate]

class Quiz(QuizBase):
    id: int
    questions: List[Question]

    class Config:
        orm_mode = True

class CourseCreate(BaseModel):
    course_name: str

class Course(CourseCreate):
    id: int
    user_id: int

    class Config:
        orm_mode = True

class CourseReturn(BaseModel):
    id: int
    course_name: str

    class Config:
        orm_mode = True

class reviewKeyConcepts(ReviewBase):
    keyConcepts: List[KeyConceptReviewBase]