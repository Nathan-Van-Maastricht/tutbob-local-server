
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import tempfile
import json
import openai
from tenacity import retry, stop_after_attempt, wait_fixed
import os 
import re
from deepgram import Deepgram
from pydantic import BaseModel
from typing import Generator, List
import logging

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.security.oauth2 import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
import os
from sqlalchemy.orm import Session
import models, schemas, database
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()



#DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

ALGORITHM = "HS256"  # Specifying the signing algorithm

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")






openai.api_key = OPENAI_API_KEY
EMBEDDING_MODEL = "text-embedding-3-small"

class Transcript(BaseModel):
    transcript: str

class TextConcept(BaseModel):
    text: str
    concept: str

class Chat(BaseModel):
    question: str
    concept: str


app = FastAPI()

# Set up CORS for the FastAPI app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # allows all methods
    allow_headers=["*"],  # allows all headers
)

@app.get("/")
def root():
    return "If not me, then who?"

async def transcribe_audio(audio_file: UploadFile, deepgram_api_key=DEEPGRAM_API_KEY):
    audio_data = await audio_file.read()
    
    # Save the WebM audio data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm_file:
        temp_webm_file.write(audio_data)
        temp_webm_file.flush()

        # Initialize the Deepgram SDK
        deepgram = Deepgram(deepgram_api_key)

        # Open the audio file
        audio = open(temp_webm_file.name, 'rb')

        # Set the source
        source = {
            'buffer': audio,
            'mimetype': 'audio/webm'
        }

        # Send the audio to Deepgram and get the response
        response = await deepgram.transcription.prerecorded(
            source,
            {
                'punctuate': True,
                'detect_language': True,
                'model': 'nova',
            }
        )

        os.remove(temp_webm_file.name)

        return response["results"]["channels"][0]["alternatives"][0]["transcript"]
    
    

@app.post("/findCurrentConceptFunc")
def findCurrentConceptFunc(transcript: Transcript):
    try:
        concept =  current_concept_func(transcript.transcript)
        if concept:
            return {"concept": concept}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to identify key concepts, output: {concept}")
    except Exception as e:
        logging.exception("An error occurred while updating key concepts")

@retry(stop=stop_after_attempt(5), wait=wait_fixed(15))
def current_concept_func(transcriptions: str):
    
    example_output = {
        "heading": "Machine Learning",
        "explanation": "Machine Learning is a type of artificial intelligence that provides computers the ability to learn without being explicitly programmed."
    }

    messages = [
        {"role": "system", "content": "You are an AI tutor that helps students understand content from lecture transcripts."},
        {"role": "user", "content": "Given the following lecture transcript: \"" + transcriptions + "\", please provide a detailed explanation of the main concept. The transcript is AI generated, so some words might be mispelled. Try your best to understand from the context. Your response should be a JSON object that includes one 'heading' for the main concept and one 'explanation'. Here's an example of the expected format: " + json.dumps(example_output)}
    ]
    
    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        n=1,
        request_timeout=15,
        stop=None,
        temperature=0.1
    )

    if response.choices:
        key_concepts_str = response.choices[0].message['content'].strip()
        print(key_concepts_str)
        key_concepts_str = re.sub(r',\s*}', '}', key_concepts_str)
        key_concepts_str = key_concepts_str.replace('\n', '\\n') 
        key_concepts = json.loads(key_concepts_str)
        return key_concepts

    else:
        return None



@app.post("/updateKeyConcepts")
def update_key_concepts(transcript: Transcript):
    try:
        key_concepts =  identify_key_concepts(transcript.transcript)
        if key_concepts:
            print(key_concepts)
            return key_concepts
        else:
            raise HTTPException(status_code=400, detail=f"Failed to identify key concepts, output: {key_concepts}")
    except Exception as e:
        logging.exception("An error occurred while updating key concepts")


@retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
def identify_key_concepts(transcriptions: str):
    example_output = {
    "summary" : "This section of the lecture is about Machine Learning and Artificial Intelligence...",
    "key_concepts": {
        'Machine Learning': 'Machine Learning is a type of artificial intelligence that provides computers the ability to learn without being explicitly programmed.',
        'Artificial Intelligence': 'Artificial Intelligence is a branch of computer science dealing with the simulation of intelligent behavior in computers.'
    }
}


    messages = [
    {"role": "system", "content": "You are an AI tutor that identifies and explains key concepts from lectures."},
    {"role": "user", "content": f"Provide a summary and identify the key concepts that are covered in the following section of a lecture : {transcriptions}. Format as a JSON object with the concepts as keys and explanations as values. Each concept and explanation should be enclosed in quotes and separated by a comma, like so: {json.dumps(example_output)}."},
]

    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        n=1,
        stop=None,
        temperature=0.1
    )

    if response.choices:
        key_concepts_str = response.choices[0].message['content'].strip()
        key_concepts_str = re.sub(r',\s*}', '}', key_concepts_str)
        try:
            key_concepts = json.loads(key_concepts_str)
            return key_concepts
        except json.JSONDecodeError:
            newMessage = [
                {"role": "user", "content": f"Please format the following into valid JSON format: {key_concepts_str}. Use the following format as an example to format this response: {json.dumps(example_output)}"}
            ]
            formatJSON = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=newMessage,
                n=1,
                stop=None,
                temperature=0.1
            )
            if formatJSON.choices:
                key_concepts_str = formatJSON.choices[0].message['content'].strip()
                key_concepts_str = re.sub(r',\s*}', '}', key_concepts_str)
                key_concepts = json.loads(key_concepts_str)
                return key_concepts
    else:
        return None




def process_text_stream(text: str, action: str, concept: str) -> Generator[str, None, None]:
    prompt = ""
    if action == "simplify":
        prompt = f"Provide a short and simple explanation for the concept: {concept} based on this text: {text}. Keep it under 3 sentences."
    else: prompt = f"{action.capitalize()} the following explanation: {text}, for concept: {concept}"
    
    messages = [
        {"role": "system", "content": f"{action} the explanation."},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        n=1,
        stop=None,
        temperature=0,
        stream=True
    )
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            message = f"data: {json.dumps({'delta': delta['content']})}\n\n"
            yield message
    yield "event: complete\ndata: {}\n\n"


@app.post("/simplify")
def simplify_text(text_concept: TextConcept):
    sseURL = f"/stream/simplify/{text_concept.text}/{text_concept.concept}"
    return {"url": sseURL}

@app.post("/elaborate")
def elaborate_text(text_concept: TextConcept):
    sseURL = f"/stream/elaborate/{text_concept.text}/{text_concept.concept}"
    return {"url": sseURL}

@app.get("/stream/{action}/{text}/{concept}")
def text_process(action: str, text: str, concept: str):
    stream = process_text_stream(text, action, concept)
    return StreamingResponse(stream, media_type="text/event-stream")

def process_chat_stream(question: str, concept: str) -> Generator[str, None, None]:
    prompt = f"Answer the following question related to {concept}: {question}"
    messages = [
        {"role": "system", "content": "Answer the question."},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        n=1,
        stop=None,
        temperature=0,
        stream=True
    )
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            message = f"data: {json.dumps({'response': delta['content']})}\n\n"
            yield message
    yield "event: complete\ndata: {}\n\n"

@app.post("/chat")
def chat(chat: Chat):
    # Streaming response URL is /chat_stream/<question>/<concept>
    sseURL = f"/chat_stream/{chat.question}/{chat.concept}"
    return {"url": sseURL}

@app.get("/chat_stream/{question}/{concept}")
def chat_process(question: str, concept: str):
    stream = process_chat_stream(question, concept)
    return StreamingResponse(stream, media_type="text/event-stream")


@app.post("/createNotes")
def create_notes(transcript: Transcript):
    notes, heading, concepts =  generate_notes(transcript.transcript)
    if notes and heading and concepts:
        return {"notes": notes, "heading": heading, "concepts": concepts}
    else:
        raise HTTPException(status_code=400, detail="Failed to generate notes")

@retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
def generate_notes(transcript: str):
    example_output = {
        "heading": "Machine Learning and Artificial Intelligence",
        "notes": [
            "- Definition: Machine Learning is a type of artificial intelligence...",
            "- Types: There are three types of Machine Learning...",
            "- ...",
        ],
        "concepts": ["Machine Learning", "Supervised Learning", "Unsupervised Learning"]
    }

    messages = [
        {"role": "user", "content": f"Create notes for university students from the following extract of a lecture transcript: {transcript}. The transcript is AI generated, so some words might be mispelled, try your best to understand from the context. Keep the notes concise and only focus on the most important points. Please format your response as a JSON object with keys 'heading', 'notes', and 'concepts'. Each note should start with a bullet point. The format should look like this: {json.dumps(example_output)}."}
    ]
    
    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        n=1,
        stop=None,
        request_timeout=30,
        temperature=0.2
    )

    if response.choices:
        try:
            output = json.loads(response.choices[0].message.content.strip())
            notes = output.get("notes")
            heading = output.get("heading")
            concepts = output.get("concepts")
            if notes and heading and concepts:
                return notes, heading, concepts
            else:
                raise ValueError("Incomplete output")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON output")
    else:
        raise ValueError("No response from model")



@app.post("/createQuiz")
def create_quiz(transcript: Transcript):
    questions =  generate_quiz(transcript.transcript)
    if questions:
        return questions
    else:
        raise HTTPException(status_code=400, detail="Failed to generate quiz questions")

@retry(stop=stop_after_attempt(5), wait=wait_fixed(15))
def generate_quiz(transcript: str):
    example_output = {
        "quiz": {
            "question1": {
                "question": "What is machine learning?",
                "answers": {"a": "The study of computer algorithms", 
                            "b": "A branch of artificial intelligence", 
                            "c": "The use of statistical methods", 
                            "d": "All of the above"},
                "correct_answer": "d",
                "explanation": "Machine learning is a branch of artificial intelligence that uses statistical methods to enable machines to improve with experience."
            }
        }
    }
    
    messages = [
        {"role": "system", "content": "You are an AI assistant that can create a multiple-choice quiz based on provided lecture transcripts."},
        {"role": "user", "content": f"Create a multiple-choice quiz with 2 to 3 questions from the following lecture transcript: {transcript}. For each question, include the question, answers, correct answer, and an explanation of why the answer is correct. Format the output as a JSON object in the following structure: {json.dumps(example_output)}"}
    ]
    
    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        n=1,
        stop=None,
        temperature=0
    )

    if response.choices:
        quiz_str = response.choices[0].message['content'].strip()
        quiz = json.loads(quiz_str)
        return quiz
    else:
        return None
    

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/signup", response_model=schemas.Token)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = pwd_context.hash(user.password)
    db_user = models.User(username=user.username, password_hash=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    token_data = {"sub": str(db_user.id)}
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/login", response_model=schemas.Token)
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()

    # Check if user exists and password is correct
    if not db_user or not pwd_context.verify(user.password, db_user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
        )

    # User exists and the password is correct, so we create a new token and return it
    token_data = {"sub": str(db_user.id)}
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}



def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


async def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        print(f"Token: {token}")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
        user_id: int = int(payload.get("sub"))
        if user_id is None:
            raise credentials_exception
        else: 
            print(user_id)

    except Exception as e:
        print(f"Error decoding JWT: {e}")
        raise credentials_exception
    
    user = get_user_by_id(db, user_id=user_id)
    if user is None:
        raise credentials_exception
    else: 
        print(user)
    return user


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...),
                         db: Session = Depends(get_db),
                         current_user: models.User = Depends(get_current_user)
                         ):
    if current_user.remaining_seconds < 15:
        raise HTTPException(status_code=400, detail="Not enough remaining transcription time")
    
    transcription = await transcribe_audio(audio)
    current_user.remaining_seconds -= 15
    db.commit()
    if transcription:
        return {"transcription": transcription}
    else:
        raise HTTPException(status_code=400, detail="Failed to transcribe audio")
    

def create_lecture(db: Session, dateTime: datetime, user_id: int):
    db_lecture = models.Lecture(user_id=user_id, date_time=dateTime)
    db.add(db_lecture)
    db.commit()
    db.refresh(db_lecture)
    return {"id": db_lecture.id}

@app.post("/lectures", response_model=schemas.LectureID)
async def create_lecture_for_user(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    db_lecture = create_lecture(db=db, dateTime = datetime.utcnow(), user_id=current_user.id)
    return db_lecture

def create_note(db: Session, heading: str, notes: str, lecture_id: int):
    db_note = models.Note(heading=heading, notes=notes, lecture_id=lecture_id)
    db.add(db_note)
    db.commit()
    db.refresh(db_note)

@app.post("/notes")
async def notes_route(
    note: schemas.NoteCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_lecture = db.query(models.Lecture).filter(models.Lecture.id == note.lecture_id).first()
    if not db_lecture:
        raise HTTPException(status_code=401, detail="Lecture not found")
    if db_lecture.user_id != current_user.id:
        raise HTTPException(status_code=401, detail="Access to Lecture not authorized")
    create_note(db=db, heading=note.heading, notes=note.notes, lecture_id=note.lecture_id)

def create_live_transcription(db: Session, transcript_text: str, timestamp: datetime, lecture_id: int):
    db_live_transcription = models.RealTimeTranscript(transcript_text=transcript_text, timestamp=timestamp, lecture_id=lecture_id)
    db.add(db_live_transcription)
    db.commit()
    db.refresh(db_live_transcription)

@app.post("/saveLiveTranscript")
async def saveLiveTranscript(
    live_transcript: schemas.RealTimeTranscriptCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)):

    db_lecture = db.query(models.Lecture).filter(models.Lecture.id == live_transcript.lecture_id).first()
    if not db_lecture:
        raise HTTPException(status_code=401, detail="Lecture not found")
    if db_lecture.user_id != current_user.id:
        raise HTTPException(status_code=401, detail="Access to Lecture not authorized")
    create_live_transcription(db=db, transcript_text=live_transcript.transcript_text, timestamp=datetime.utcnow(), lecture_id = live_transcript.lecture_id)

def create_review(db: Session, interval: str, summary: str, lecture_id: int):
    db_review = models.Review(interval=interval, summary=summary, lecture_id=lecture_id)
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

@app.post("/saveReview")
async def saveReview(
    review: schemas.ReviewCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)):
    print(review)

    db_lecture = db.query(models.Lecture).filter(models.Lecture.id == review.lecture_id).first()
    if not db_lecture:
        raise HTTPException(status_code=401, detail="Lecture not found")
    if db_lecture.user_id != current_user.id:
        raise HTTPException(status_code=401, detail="Access to Lecture not authorized")
    db_review = create_review(db=db, interval=review.interval, summary=review.summary, lecture_id= review.lecture_id)

    for key_concept in review.key_concepts:
        db_key_concept = models.KeyConcept(
            review_id=db_review.id,
            concept_text=key_concept.concept_text,
            explanation_text=key_concept.explanation_text,
            simplify_text=key_concept.simplify_text,
            elaborate_text=key_concept.elaborate_text,
        )
        db.add(db_key_concept)
    db.commit()

@app.post("/saveQuiz")
def create_quiz(
    quiz: schemas.QuizCreate, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)):

    db_lecture = db.query(models.Lecture).filter(models.Lecture.id == quiz.lecture_id).first()
    if not db_lecture:
        raise HTTPException(status_code=401, detail="Lecture not found")
    if db_lecture.user_id != current_user.id:
        raise HTTPException(status_code=401, detail="Access to Lecture not authorized")

    # Create a new quiz
    db_quiz = models.Quiz(lecture_id=quiz.lecture_id)
    db.add(db_quiz)
    db.commit()
    db.refresh(db_quiz)

    # Create questions and associate them with the quiz
    for question_data in quiz.questions:
        db_question = models.Question(
            quiz_id=db_quiz.id,
            question_text=question_data.question,
            explanation=question_data.explanation,
            correct_answer=question_data.correct_answer
        )
        db.add(db_question)
        db.commit()
        db.refresh(db_question)

        # Create answers and associate them with the question
        for answer in question_data.answers:
            db_answer = models.Answer(
                question_id=db_question.id,
                answer_text=answer.answer_text,
                answer_option=answer.answer_option
            )
            db.add(db_answer)
        db.commit()

    db.refresh(db_quiz)


@app.get("/getCourses", response_model=List[schemas.CourseReturn])
def getCourse(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Query the database to get courses for the current user
    courses = db.query(models.Course).filter(models.Course.user_id == current_user.id).all()

    return courses

@app.post("/createCourse")
def createCourse(
    course: schemas.CourseCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # First, create a new Course model instance
    new_course = models.Course(
        user_id=current_user.id, 
        course_name=course.course_name
    )

    # Add the new course to the session
    db.add(new_course)
    
    # Commit the changes to the database
    db.commit()

    # Refresh the instance to get the newly generated ID
    db.refresh(new_course)


@app.post("/updateLecture")
def updateLecture(
    lecture: schemas.LectureUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_lecture = db.query(models.Lecture).filter(models.Lecture.id == lecture.id).first()
    if not db_lecture:
        raise HTTPException(status_code=401, detail="Lecture not found")
    if db_lecture.user_id != current_user.id:
        raise HTTPException(status_code=401, detail="Access to Lecture not authorized")
    
    db_lecture.title = lecture.title
    db_lecture.course_id = lecture.course_id

    # Commit the changes to the database
    db.commit()

    # Refresh the lecture object to get the updated data from the database
    db.refresh(db_lecture)
    

@app.post("/populateCourses", response_model=List[schemas.CourseWithLectures])
def populateCourse(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    result = db.query(models.Course.id, models.Course.course_name, models.Lecture.id, models.Lecture.title, models.Lecture.date_time)\
        .outerjoin(models.Lecture, models.Course.id == models.Lecture.course_id)\
        .filter(models.Course.user_id == current_user.id).all()
    
    structured_results = {}
    for course_id, course_name, lecture_id, lecture_title, lecture_date in result:
        if course_id not in structured_results:
            structured_results[course_id] = {
                "course_name": course_name,
                "lectures": []
            }
        structured_results[course_id]["lectures"].append({
            "id": lecture_id,
            "title": lecture_title,
            "date_time": lecture_date
        })
    
    output = [schemas.CourseWithLectures(course_name=data["course_name"], course_id=course_id, lectures=data["lectures"]) for course_id, data in structured_results.items()]
    return output


@app.get("/populateReview/{lecture_id}", response_model=List[schemas.reviewKeyConcepts])
def populateReview(
    db: Session = Depends(get_db),
    lecture_id: int = None
):
    result = db.query(models.Review.interval, models.Review.summary, models.KeyConcept.concept_text, models.KeyConcept.explanation_text)\
        .join(models.KeyConcept, models.Review.id == models.KeyConcept.review_id)\
        .filter(models.Review.lecture_id == lecture_id).all()
    
    structured_results = {}
    for interval, summary, concept_text, explanation_text in result:
        if interval not in structured_results:
            structured_results[interval] = {
                "interval": interval,
                "summary": summary,
                "keyConcepts": []
            }
        structured_results[interval]["keyConcepts"].append({
            "concept_text": concept_text,
            "explanation_text": explanation_text
        })
    output = [schemas.reviewKeyConcepts(interval=data["interval"], summary=data["summary"], keyConcepts=data["keyConcepts"]) for _, data in structured_results.items()]
    return output

@app.get("/populateNotes/{lecture_id}", response_model=List[schemas.NoteBase])
def populateReview(
    db: Session = Depends(get_db),
    lecture_id: int = None
):
    result = db.query(models.Note.heading, models.Note.notes).filter(models.Note.lecture_id == lecture_id).all()
    
    output : List[schemas.NoteBase] = []

    for heading, notes in result:
        output.append(schemas.NoteBase(heading=heading, notes=notes))
    return output

@app.get("/populateQuiz/{lecture_id}", response_model=List[schemas.QuestionCreate])
def populateReview(
    db: Session = Depends(get_db),
    lecture_id: int = None
):
    quiz_id = db.query(models.Quiz.id).filter(models.Quiz.lecture_id == lecture_id).scalar()
    result = db.query(models.Question.question_text, models.Question.explanation, models.Question.correct_answer, models.Answer.answer_text, models.Answer.answer_option)\
        .join(models.Answer, models.Question.id == models.Answer.question_id)\
        .filter(models.Question.quiz_id == quiz_id).all()
    
    structured_results = {}
    for question, explanation, correct_answer, answer_text, answer_option in result:
        if question not in structured_results:
            structured_results[question] = {
                "question": question,
                "explanation": explanation,
                "correct_answer" : correct_answer,
                "answers": []
            }
        structured_results[question]["answers"].append({
            "answer_text": answer_text,
            "answer_option": answer_option
        })
    output = [schemas.QuestionCreate(question=data["question"], explanation=data["explanation"], correct_answer=data["correct_answer"], answers=data["answers"]) for _, data in structured_results.items()]
    return output


def create_live_transcription(db: Session, transcript_text: str, timestamp: datetime, lecture_id: int):
    db_live_transcription = models.RealTimeTranscript(transcript_text=transcript_text, timestamp=timestamp, lecture_id=lecture_id)
    db.add(db_live_transcription)
    db.commit()
    db.refresh(db_live_transcription)

@app.post("/saveLiveTranscript")
async def saveLiveTranscript(
    live_transcript: schemas.RealTimeTranscriptCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)):

    db_lecture = db.query(models.Lecture).filter(models.Lecture.id == live_transcript.lecture_id).first()
    if not db_lecture:
        raise HTTPException(status_code=401, detail="Lecture not found")
    if db_lecture.user_id != current_user.id:
        raise HTTPException(status_code=401, detail="Access to Lecture not authorized")
    create_live_transcription(db=db, transcript_text=live_transcript.transcript_text, timestamp=datetime.utcnow(), lecture_id = live_transcript.lecture_id)

@app.post("/addEmbeddingVector")
async def addEmbeddingVector(
    baseVector: schemas.VectorEmbeddingBase,
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)):

        db_lecture = db.query(models.Lecture).filter(models.Lecture.id == baseVector.lecture_id).first()
        if not db_lecture:
            raise HTTPException(status_code=401, detail="Lecture not found")
        if db_lecture.user_id != current_user.id:
            raise HTTPException(status_code=401, detail="Access to Lecture not authorized")

        vector_json = create_new_embedding(baseVector.transcript)
        if(vector_json):
            save_embedding(db=db, embedding=vector_json, lecture_id=baseVector.lecture_id, transcript=baseVector.transcript)

def save_embedding(db: Session, embedding: str, lecture_id: int, transcript:str):
    db_embedding = models.VectorEmbedding(vector=embedding, lecture_id=lecture_id, transcript=transcript)
    db.add(db_embedding)
    db.commit()
    db.refresh(db_embedding)

def create_new_embedding(transcript):
    response = openai.Embedding.create(
        input = transcript,
        model = EMBEDDING_MODEL
    )
    if response.data:
        return json.dumps(response.data[0].embedding)
    
    return None
def get_stored_embeddings(db: Session, lecture_id: int):
    # Fetch the embeddings from the database
    db_embeddings = db.query(models.VectorEmbedding).filter(models.VectorEmbedding.lecture_id == lecture_id).all()
    if db_embeddings:
    # Convert each JSON string embedding to a list of floats
        embeddings = [json.loads(embedding.vector) for embedding in db_embeddings]
        text = [embedding.transcript for embedding in db_embeddings]

        return embeddings, text
    
    return None

def answer_question(question, transcriptArray):
    messages = [
        {"role": "user", "content": f"Answer the following question: {question} to the best of your ability. Where possible, use the following source as a reference: {transcriptArray}."}
    ]
    
    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages,
        n=1,
        request_timeout=15,
        stop=None,
        temperature=0
    )

    if response.choices:
        return response.choices[0].message['content'].strip()

@app.post("/ask_question_embedding")
async def ask_question_embedding(
    question: schemas.VectorEmbeddingBase,
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)):

        db_lecture = db.query(models.Lecture).filter(models.Lecture.id == question.lecture_id).first()
        if not db_lecture:
            raise HTTPException(status_code=401, detail="Lecture not found")
        if db_lecture.user_id != current_user.id:
            raise HTTPException(status_code=401, detail="Access to Lecture not authorized")

        (stored_embeddings, transcript) = get_stored_embeddings(db=db, lecture_id=question.lecture_id)
        if stored_embeddings:
            question_embedding = create_new_embedding(question.transcript)
            if question_embedding:
                question_embedding = json.loads(question_embedding)
            else:
                raise HTTPException(status_code=400, detail="Failed to create embedding for question")
            
            if question_embedding:
                similarity_scores = []
                for embedding in stored_embeddings:
                    similarity_score = cosine_similarity(np.array(embedding).reshape(1, -1), np.array(question_embedding).reshape(1, -1))[0][0]
                    similarity_scores.append(similarity_score)

                    top_indices = np.argsort(similarity_scores)[-3:][::-1]  # This gets the indices of the top 3 scores
                print(max(similarity_scores))

                transcriptArray = [transcript[index] for index in top_indices]
                print(transcriptArray)
                return answer_question(question.transcript, transcriptArray)



            else:
                raise HTTPException(status_code=401, detail="Failed to create embedding for question")
        else:
            raise HTTPException(status_code=402, detail="No stored embeddings for lecture")


