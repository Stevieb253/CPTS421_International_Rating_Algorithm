# Sprint 5 Report (2/17/2026-3/15/2026)

## YouTube link of Sprint * Video [youtube.com](https://youtu.be/Ap_DkOtFV-E)

## What's New (User Facing)  
* Report generation bug fixed from end to end  
* Full WSU-branded UI across all pages  
* Analytics dashboard now updates in real-time  
* Admin panel working Deactivate & Delete buttons  
* Risk factor checkboxes now multi-select correctly

## Work Summary (Developer Facing)

- Khushi resolved a chain of backend bugs blocking report generation three stacked root causes (wrong return type, DB constraint, numpy types in ReportLab) each hiding the others. The database layer expanded with new methods. All templates rebuilt against the official WSU design system required a fully custom checkbox implementation after the WSU bundle broke native rendering JS TypeError was blocking the transcript analyze button, with no visible error diagnosed by auditing every top-level script statement. Steven added a loading bar to both fraud detection pages, built an automated test suite for both fraud systems, and fixed text overflow in fraud PDF exports. The team began configuring Render deployment for the upcoming client demo.

## Unfinished Work

- Analysis results reset on server restart persisting to DB deprioritized after bug fixes exceeded time estimates.  
- Render deployment not yet live HuggingFace token configuration still in progress report_type column not added to DB deprioritized in favor of blocking bugs.  
- Student detail page missing analyst name and recommendation requires schema changes, deferred to Sprint 6.  
- More system-wide unit testing.

## Completed Issues/User Stories  
Here are links to the issues that we completed in this sprint:  
* https://github.com/Stevieb253/CPTS421\_International\_Rating\_Algorithm/issues/40  
* https://github.com/Stevieb253/CPTS421\_International\_Rating\_Algorithm/issues/39  
* https://github.com/Stevieb253/CPTS421\_International\_Rating\_Algorithm/issues/41

    
## Incomplete Issues/User Stories  
Here are links to issues we worked on but did not complete in this sprint:  
* https://github.com/Stevieb253/CPTS421\_International\_Rating\_Algorithm/issues/42 \<\<Ran out of time while working on other issues\>\>

    
## Code Files for Review  
Please review the following code files, which were actively developed during this  
sprint, for quality:  
* [app.py](https://github.com/your_repo/code/app.py)  
* [db/database.py](https://github.com/your_repo/code/db/database.py)   
* [db/report\_generator.py](https://github.com/your_repo/code/db/report_generator.py)   
* [templates/index.html](https://github.com/your_repo/code/templates/index.html)   
* [templates/financial.html](https://github.com/your_repo/code/templates/financial.html)   
* [templates/transcript.html](https://github.com/your_repo/code/templates/transcript.html)  
    
## Retrospective Summary  
Here's what went well:  
* Systematic debugging meant every fix was permanent, not a workaround  
* WSU CDN bundle saved significant time on the redesign  
* Steven's test suite adds reliability coverage the project didn't have before  
* Auto-migrating DB means zero manual setup on deployment

Here's what we'd like to improve:  
* Frontend scope was underestimated all templates took as long as all backend fixes  
* Silent JS errors need better client-side logging  
* UI work should be split across both team members next sprint

Here are changes we plan to implement in the next sprint:  
* Expand test suite to cover analysis and report routes

