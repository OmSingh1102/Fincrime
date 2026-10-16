"""Straive KYC Operations Hub: a second, Straive-branded demo over the
1,000-case reconciliation log, with AI agents and a human in the loop.

Package map:
  loader / parse     the CSV becomes case records (every field parsed, nothing judged)
  names / addresses  rule tables for the Identity Normalisation Agent
  similarity         folding, Jaro-Winkler, phonetic keys
  documents          synthetic uploads: a deterministic defect plan and SVG rendering
  agents/            Document Intake, Identity Normalisation, Watchlist Adjudication,
                     Triage & Routing (deterministic), plus the Claude agents on demand
  pipeline           settings and the run over the whole book
  actions            what an analyst does to a case
  analytics          the view-models the screens read
  state / views      the in-memory book, its persistence, and the Flask blueprint
"""
