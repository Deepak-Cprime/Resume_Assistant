"""Pydantic models for everything extracted from one resume."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Contact(BaseModel):
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    location: str = ""


class Experience(BaseModel):
    title: str = ""
    company: str = ""
    location: str = ""
    start: str | None = None  # "2021-03", or "2018" when the resume gives only a year
    end: str | None = None  # None while is_current
    is_current: bool = False
    bullets: list[str] = Field(default_factory=list)
    description: str = ""
    # high: exactly one side of the header looks like a job title; low: guessed order.
    confidence: Literal["high", "low", "none"] = "none"
    source: Literal["rules", "llm"] = "rules"
    raw_text: str = ""


class Education(BaseModel):
    degree: str = ""
    school: str = ""
    year: int | None = None


class Certification(BaseModel):
    name: str
    year: int | None = None


class SkillEvidence(BaseModel):
    skill: str
    claimed: bool = False  # listed in the skills or summary section
    used: bool = False  # appears in an experience entry
    years: float | None = None  # time covered by the jobs that mention it

    @property
    def evidence(self) -> str:
        return "used" if self.used else "claimed"


class Gap(BaseModel):
    start: str
    end: str
    months: int


class SectionInfo(BaseModel):
    label: str
    heading: str = ""
    method: str
    text: str


class CandidateProfile(BaseModel):
    name: str = ""
    email_hash: str = ""
    contact: Contact = Field(default_factory=Contact)
    headline: str = ""
    seniority: str = ""
    total_yoe: float = 0.0
    gaps: list[Gap] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[SkillEvidence] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    sections: list[SectionInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
