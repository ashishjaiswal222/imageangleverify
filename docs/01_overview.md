# 1. Overview & Problem Solved

## The Problem
In highly secure onboarding environments (such as KYC - Know Your Customer), verifying a user's identity accurately requires strict compliance regarding photo quality, angles, and identity consistency. 
Without automated verification, users often upload photos that are blurry, poorly lit, heavily edited, or lacking the required visibility (e.g., wearing sunglasses or submitting group photos). Furthermore, bad actors may attempt to bypass security by submitting photos of different people across different required angles.

## The Solution
The **Photo Verification API** is an intelligent, offline-first microservice designed to act as an automated gatekeeper. 

It solves the problem by providing instantaneous feedback on photo quality and angle accuracy through deeply integrated Computer Vision (CV) heuristics. It completely automates:
- **Quality Assurance**: Rejecting blurry, poorly lit, or occluded photos.
- **Angle & Pitch Verification**: Mathematically proving the user has submitted the correct views (Front, Left Profile, Right Profile, Full Body, and Back), while enforcing strict eye-level camera constraints.
- **AI-Grade Consistency**: Enforcing uncluttered backgrounds and neutral facial expressions to ensure photos are perfectly suitable as reference material for AI image generation.
- **Fraud Prevention**: Extracting 128-dimensional facial biometrics to perform cross-photo identity verification, guaranteeing that the person in the Front photo is the exact same person in the Left, Right, and Full Body photos.
- **Explicit Content Blocking**: Preventing the upload of NSFW images locally without third-party APIs.

### Comprehensive Failure Reporting
Unlike traditional APIs that stop at the first error, this API performs a full-pass scan and returns a comprehensive `failed_reasons` array. This allows users to fix all issues (e.g. lighting, background, and expression) in a single attempt rather than being frustrated by multiple re-uploads.

This ensures the backend system only receives high-quality, verified, and consistent datasets, drastically reducing manual review time.
