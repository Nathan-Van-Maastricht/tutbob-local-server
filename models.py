from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "Users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    password_hash = Column(String(255))
    remaining_seconds = Column(Integer, default=0)  
    lectures = relationship("Lecture", back_populates="user")

class Lecture(Base):
    __tablename__ = "Lectures"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('Users.id'))
    course_id = Column(Integer, ForeignKey('Courses.id'), nullable=True)
    title = Column(String(100), nullable=True)
    date_time = Column(DateTime)

    # this will allow you to access the owner of a lecture by lecture.user
    user = relationship("User", back_populates="lectures")
    notes = relationship("Note", back_populates="lecture")
    real_time_transcripts = relationship("RealTimeTranscript", back_populates="lecture")
    reviews = relationship("Review", back_populates="lecture")
    course = relationship("Course", back_populates="lectures")
    vector_embeddings = relationship("VectorEmbedding", back_populates="lecture")


class VectorEmbedding(Base):
    __tablename__ = "VectorEmbeddings"

    id = Column(Integer, primary_key=True, index=True)
    lecture_id = Column(Integer, ForeignKey("Lectures.id"))
    vector = Column(Text) 
    transcript = Column(Text) 

    lecture = relationship("Lecture", back_populates="vector_embeddings")


class Note(Base):
    __tablename__ = "Notes"

    id = Column(Integer, primary_key=True, index=True)
    lecture_id = Column(Integer, ForeignKey("Lectures.id"))
    heading = Column(String(255))
    notes = Column(Text)

    # Relationship to Lecture model
    lecture = relationship("Lecture", back_populates="notes")


class KeyConcept(Base):
    __tablename__ = "KeyConcepts"

    id = Column(Integer, primary_key=True, index=True)
    review_id = Column(Integer, ForeignKey("Reviews.id"))
    concept_text = Column(Text)
    explanation_text = Column(Text)
    simplify_text = Column(Text)
    elaborate_text = Column(Text)

    review = relationship("Review", back_populates="key_concepts")

class RealTimeTranscript(Base):
    __tablename__ = "RealTimeTranscripts"

    id = Column(Integer, primary_key=True, index=True)
    lecture_id = Column(Integer, ForeignKey("Lectures.id"))
    transcript_text = Column(String)
    timestamp = Column(DateTime)

    lecture = relationship("Lecture", back_populates="real_time_transcripts")

class Review(Base):
    __tablename__ = 'Reviews'

    id = Column(Integer, primary_key=True, index=True)
    lecture_id = Column(Integer, ForeignKey('Lectures.id'))
    interval = Column(String(50), nullable=False)
    summary = Column(String, nullable=False)

    
    lecture = relationship("Lecture", back_populates="reviews")
    key_concepts = relationship("KeyConcept", back_populates="review")

class Quiz(Base):
    __tablename__ = 'Quizzes'
    
    id = Column(Integer, primary_key=True)
    lecture_id = Column(Integer, nullable=True)

    questions = relationship('Question', back_populates='quiz')

class Question(Base):
    __tablename__ = 'Questions'
    
    id = Column(Integer, primary_key=True)
    quiz_id = Column(Integer, ForeignKey('Quizzes.id'))
    question_text = Column(String, nullable=True)
    explanation = Column(String, nullable=True)
    correct_answer = Column(String(1), nullable=True)

    quiz = relationship('Quiz', back_populates='questions')
    answers = relationship('Answer', back_populates='question')

class Answer(Base):
    __tablename__ = 'Answers'
    
    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey('Questions.id'))
    answer_text = Column(String, nullable=True)
    answer_option = Column(String(1), nullable=True)

    question = relationship('Question', back_populates='answers')

class Course(Base):
    __tablename__ = 'Courses'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('Users.id'))
    course_name = Column(String, nullable=False)
    lectures = relationship("Lecture", back_populates="course")