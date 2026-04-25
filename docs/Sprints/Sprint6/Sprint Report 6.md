# Sprint 6 Report (03/16/2026 – 04/24/2026)

## YouTube link of Sprint 6 Video (Make this video unlisted)
*[Click here to access the link](https://youtu.be/HQUHaxc2waY)*

## What's New (User Facing)
* Created a 3-minute professional video showcasing IARA.
* Created a final poster presentation.
* Created a final report document for the stakeholder.
* Created a hand-over document for the stakeholder.
* Deployed application to Azure App Service / Azure Web App.
* Prepared the project for stakeholder delivery and future continuation.

## Work Summary (Developer Facing)
During this sprint, our team focused on finalizing the IARA system and preparing it for delivery to the stakeholder. Instead of adding major new features, most of the work centered on integration, deployment, and ensuring all components worked reliably together. We connected the scoring system, NLP analysis, and fraud detection modules through the Flask backend and verified consistent outputs across the application. One of the main challenges we faced was deployment. Our initial attempt using a lightweight hosting platform ran into memory limitations, which caused instability when running more intensive features like document processing. To address this, we transitioned to Azure App Service and adjusted our configuration to better support the application’s resource needs. This required setting up environment variables, handling dependencies, and ensuring the system could run in a hosted environment without breaking existing functionality. We also improved overall stability, including fixing issues with report generation and making sure the system handled real user inputs correctly. This sprint reinforced the importance of deployment planning, system integration, and building software that works reliably outside of a local development environment.

## Unfinished Work
Since this was our final sprint, the primary focus was completing and delivering a stable, fully functional version of the IARA system rather than starting new features. Most planned work was completed; however, a few enhancements were not fully finished due to time constraints and prioritizing deployment and stakeholder deliverables. These included additional UI/UX improvements, more extensive testing (especially for multiple concurrent users), and further scalability optimizations. These items are natural next steps for the project and can be expanded on in future development as the system continues to evolve.

## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:
* [Issue 44](https://github.com/Stevieb253/CPTS421_International_Rating_Algorithm/issues/44)

## Incomplete Issues/User Stories
None

## Code Files for Review
Please review the following code files, which were actively developed during this sprint, for quality:
* Dockerfile
* .dockerignore
* code/db/database.py
* code/services/financial_fraud_detector.py
* code/services/transcript_fraud_detector.py
* requirements.txt

## Retrospective Summary

Here's what went well:
* Successfully deploying the IARA system to Azure and confirming it runs in a hosted environment.
* All core components (scoring, NLP analysis, fraud detection, and report generation) were fully integrated and working together.
* Completing all final deliverables including the video, poster, report, and handoff document.

Here's what we'd like to improve:
* Starting deployment and environment testing earlier instead of waiting until the final sprint.
* Improving communication and coordination toward the end of the sprint.

Here are changes we plan to implement in the next sprint:
* Begin deployment and scalability testing earlier in the development cycle.
* Introduce more structured testing, including unit and system-level tests.
