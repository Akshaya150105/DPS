"""Tests for preprocess.py: Enron maildir and Lucene mbox preprocessing."""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

import preprocess
from types_ import Document


# ---------------------------------------------------------------------------
# Test Helpers: Creating Fake Datasets
# ---------------------------------------------------------------------------

def create_fake_maildir(tmp_path: Path) -> Path:
    """Create a minimal fake Enron-like maildir with valid and invalid messages."""
    maildir = tmp_path / "fake_enron_maildir"
    user_a_inbox = maildir / "lay-k" / "inbox"
    user_a_sent = maildir / "lay-k" / "sent"
    user_b_inbox = maildir / "skilling-j" / "inbox"

    user_a_inbox.mkdir(parents=True, exist_ok=True)
    user_a_sent.mkdir(parents=True, exist_ok=True)
    user_b_inbox.mkdir(parents=True, exist_ok=True)

    # Message 1 (Valid date: Jan 10 2001)
    msg1 = (
        "Message-ID: <1@thyme>\n"
        "Date: Wed, 10 Jan 2001 09:00:00 -0600\n"
        "From: lay-k@enron.com\n"
        "Subject: Budget Meeting\n\n"
        "Hello team,\n"
        "We have a budget meeting scheduled for tomorrow.\n"
        "Please bring your financial reports.\n"
    )
    (user_a_inbox / "1.").write_text(msg1, encoding="utf-8")

    # Message 2 (Valid date: Jan 11 2001, contains quoted replies)
    msg2 = (
        "Message-ID: <2@thyme>\n"
        "Date: Thu, 11 Jan 2001 14:30:00 -0600\n"
        "From: skilling-j@enron.com\n"
        "Subject: Re: Budget Meeting\n\n"
        "Thanks Ken,\n"
        "I will attend the budget meeting with the quarterly forecast.\n\n"
        "> Hello team,\n"
        "> We have a budget meeting scheduled for tomorrow.\n"
        "-----Original Message-----\n"
        "Secret omitted reply block with secretword\n"
    )
    (user_b_inbox / "1.").write_text(msg2, encoding="utf-8")

    # Message 3 (Valid date: Jan 12 2001, with punctuation & short tokens)
    msg3 = (
        "Message-ID: <3@thyme>\n"
        "Date: Fri, 12 Jan 2001 16:00:00 -0600\n"
        "From: lay-k@enron.com\n"
        "Subject: Summary\n\n"
        "The meeting was great! In, at, to, we confirmed the forecast.\n"
        "Special punctuation test: budget-meeting, cost/profit!\n"
    )
    (user_a_sent / "1.").write_text(msg3, encoding="utf-8")

    # Message 4 (Bad / corrupted date)
    msg4 = (
        "Message-ID: <4@thyme>\n"
        "Date: BadDateStringHere\n"
        "From: spam@enron.com\n"
        "Subject: Corrupted\n\n"
        "This message has an invalid date and must be skipped.\n"
    )
    (user_a_inbox / "2.").write_text(msg4, encoding="utf-8")

    # Message 5 (Missing date header)
    msg5 = (
        "Message-ID: <5@thyme>\n"
        "From: spam2@enron.com\n"
        "Subject: Missing Date\n\n"
        "This message has no date header and must be skipped.\n"
    )
    (user_b_inbox / "2.").write_text(msg5, encoding="utf-8")

    return maildir


def create_fake_mbox(tmp_path: Path) -> Path:
    """Create a minimal fake Lucene mbox file."""
    mbox_dir = tmp_path / "fake_lucene_mbox"
    mbox_dir.mkdir(parents=True, exist_ok=True)
    mbox_file = mbox_dir / "lucene-dev.mbox"

    content = (
        "From user1@apache.org Tue May  2 12:00:00 2006\n"
        "Date: Tue, 02 May 2006 12:00:00 +0000\n"
        "From: user1@apache.org\n"
        "Subject: Indexing performance\n\n"
        "Lucene index optimization is improving search speed.\n\n"
        "From user2@apache.org Wed May  3 15:30:00 2006\n"
        "Date: Wed, 03 May 2006 15:30:00 +0000\n"
        "From: user2@apache.org\n"
        "Subject: Re: Indexing performance\n\n"
        "Optimization is critical for large indexes.\n"
        "> Lucene index optimization is improving search speed.\n\n"
        "From bad@apache.org Thu May  4 10:00:00 2006\n"
        "Date: InvalidDate\n"
        "From: bad@apache.org\n"
        "Subject: Broken date\n\n"
        "Should be skipped.\n\n"
    )
    mbox_file.write_text(content, encoding="utf-8")
    return mbox_dir


# ---------------------------------------------------------------------------
# Unit Tests: Text Processing and Stemming
# ---------------------------------------------------------------------------

class TestTextProcessing:
    """Tests for tokenization, punctuation removal, and reply stripping."""

    def test_remove_quoted_replies(self):
        text = (
            "Primary content.\n"
            "> Quoted line 1\n"
            "> Quoted line 2\n"
            "Still primary content.\n"
            "-----Original Message-----\n"
            "Ignored reply text.\n"
        )
        cleaned = preprocess.remove_quoted_replies(text)
        assert "Primary content." in cleaned
        assert "Still primary content." in cleaned
        assert "> Quoted line 1" not in cleaned
        assert "Ignored reply text." not in cleaned

    def test_tokenize_filters_short_tokens_and_punctuation(self):
        text = "Hello, world! We go to an AI-lab at 10am."
        tokens = preprocess.tokenize_text(text, stem=False)
        # 'we', 'go', 'to', 'an', 'at' have len < 3 and must be dropped
        for short in ["we", "go", "to", "an", "at"]:
            assert short not in tokens
        # 'hello', 'world', 'lab', '10am' (or words >= 3 chars) should remain
        assert "hello" in tokens
        assert "world" in tokens
        assert "lab" in tokens

    def test_stemming_effect(self):
        tokens_unstemmed = preprocess.tokenize_text("meetings connections", stem=False)
        tokens_stemmed = preprocess.tokenize_text("meetings connections", stem=True)
        assert "meetings" in tokens_unstemmed
        assert "connections" in tokens_unstemmed
        assert "meet" in tokens_stemmed
        assert "connect" in tokens_stemmed


# ---------------------------------------------------------------------------
# Unit Tests: Preprocessing on Fake Datasets
# ---------------------------------------------------------------------------

class TestEnronPreprocessing:
    """Tests for Enron maildir processing."""

    def test_preprocess_fake_maildir(self, tmp_path: Path, capsys):
        maildir = create_fake_maildir(tmp_path)
        out_dir = tmp_path / "processed"

        metadata = preprocess.preprocess_dataset(
            source="enron",
            path=str(maildir),
            k=500,
            stem=False,
            out_dir=str(out_dir),
        )

        # Captured output should state skipped bad dates
        captured = capsys.readouterr()
        assert "Skipped 2 messages with bad dates." in captured.out

        # Exactly 3 valid messages
        assert metadata["document_count"] == 3
        assert metadata["skipped_bad_dates"] == 2
        assert metadata["counts"]["documents"] == 3
        assert metadata["counts"]["skipped_bad_dates"] == 2

        # Verify artifacts exist
        pkl_path = out_dir / "enron.pkl"
        index_path = out_dir / "enron_index.json"
        meta_path = out_dir / "enron_metadata.json"

        assert pkl_path.exists()
        assert index_path.exists()
        assert meta_path.exists()

        # Load and verify pickle: (docs, vocab, name)
        with open(pkl_path, "rb") as f:
            docs, vocab, name = pickle.load(f)

        assert name == "enron"
        assert len(docs) == 3
        assert isinstance(vocab, list)
        assert len(vocab) > 0

        # Verify docs are valid Document instances in chronological order
        for i, doc in enumerate(docs):
            assert isinstance(doc, Document)
            assert doc.doc_id == i
            assert isinstance(doc.keywords, frozenset)

        assert docs[0].timestamp <= docs[1].timestamp <= docs[2].timestamp

        # Quoted reply word 'secretword' should NOT be in vocabulary
        assert "secretword" not in vocab

        # Verify index JSON
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        assert "meeting" in index_data
        # 'meeting' appears in all 3 documents
        assert index_data["meeting"] == [0, 1, 2]

        # Verify metadata date range
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)

        assert meta_json["date_range"]["start"] is not None
        assert meta_json["date_range"]["end"] is not None

    def test_stemmed_enron_preprocessing(self, tmp_path: Path):
        maildir = create_fake_maildir(tmp_path)
        out_dir = tmp_path / "processed_stem"

        metadata = preprocess.preprocess_dataset(
            source="enron",
            path=str(maildir),
            k=500,
            stem=True,
            out_dir=str(out_dir),
        )

        with open(out_dir / "enron.pkl", "rb") as f:
            docs, vocab, name = pickle.load(f)

        # 'meeting' should be stemmed to 'meet'
        assert "meet" in vocab
        assert "meeting" not in vocab

    def test_top_k_keyword_limit(self, tmp_path: Path):
        maildir = create_fake_maildir(tmp_path)
        out_dir = tmp_path / "processed_k2"

        metadata = preprocess.preprocess_dataset(
            source="enron",
            path=str(maildir),
            k=2,
            stem=False,
            out_dir=str(out_dir),
        )

        with open(out_dir / "enron.pkl", "rb") as f:
            docs, vocab, name = pickle.load(f)

        assert len(vocab) == 2
        for doc in docs:
            assert len(doc.keywords) <= 2
            for kw in doc.keywords:
                assert kw in vocab


class TestLucenePreprocessing:
    """Tests for Lucene mbox processing."""

    def test_preprocess_fake_mbox(self, tmp_path: Path, capsys):
        mbox_dir = create_fake_mbox(tmp_path)
        out_dir = tmp_path / "processed_lucene"

        metadata = preprocess.preprocess_dataset(
            source="lucene",
            path=str(mbox_dir),
            k=500,
            stem=False,
            out_dir=str(out_dir),
        )

        captured = capsys.readouterr()
        assert "Skipped 1 messages with bad dates." in captured.out

        assert metadata["document_count"] == 2
        assert metadata["skipped_bad_dates"] == 1

        pkl_path = out_dir / "lucene.pkl"
        assert pkl_path.exists()
        with open(pkl_path, "rb") as f:
            docs, vocab, name = pickle.load(f)

        assert name == "lucene"
        assert len(docs) == 2
        assert "optimization" in vocab or "index" in vocab


# ---------------------------------------------------------------------------
# CLI Integration Test
# ---------------------------------------------------------------------------

class TestCLI:
    """Test the command-line interface execution."""

    def test_cli_invocation(self, tmp_path: Path):
        maildir = create_fake_maildir(tmp_path)
        out_dir = tmp_path / "cli_processed"

        cmd = [
            sys.executable,
            "preprocess.py",
            "--source",
            "enron",
            "--path",
            str(maildir),
            "--k",
            "500",
            "--out-dir",
            str(out_dir),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        assert "Skipped 2 messages with bad dates." in result.stdout
        assert (out_dir / "enron.pkl").exists()
        assert (out_dir / "enron_index.json").exists()
        assert (out_dir / "enron_metadata.json").exists()


# ---------------------------------------------------------------------------
# Tests for Patch 16b: Monthly Intervals and Date Ranges
# ---------------------------------------------------------------------------

class TestMonthlyIntervals:
    """Tests for split_into_intervals and date range filtering."""

    def test_split_into_intervals_fake_maildir_lands_in_expected_months(self, tmp_path: Path):
        """Test on fake maildir that documents land in expected month intervals."""
        maildir = create_fake_maildir(tmp_path)
        out_dir = tmp_path / "processed_intervals"

        # Preprocess dataset specifying Enron default dates: 1999-12-01 to 2001-02-27
        metadata = preprocess.preprocess_dataset(
            source="enron",
            path=str(maildir),
            k=500,
            start="1999-12-01",
            end="2001-02-27",
            out_dir=str(out_dir),
        )

        with open(out_dir / "enron.pkl", "rb") as f:
            docs, vocab, name = pickle.load(f)

        assert len(docs) == 3
        # In create_fake_maildir, all 3 valid emails are from Jan 10-12, 2001.
        # From 1999-12-01, January 2001 is interval: (2001-1999)*12 + (1-12) = 13.
        for doc in docs:
            assert doc.interval == 13

        # Verify interval index is saved inside processed dataset artifacts
        assert metadata["num_intervals"] == 1
        assert metadata["intervals"] == [13]
        assert metadata["interval_index"] == {d.doc_id: 13 for d in docs}

        intervals_file = out_dir / "enron_intervals.json"
        assert intervals_file.exists()
        with open(intervals_file, "r", encoding="utf-8") as f:
            intervals_data = json.load(f)
        assert intervals_data == {str(d.doc_id): 13 for d in docs}

    def test_split_into_intervals_multi_month_and_out_of_range_dropping(self):
        """Test that documents across different months receive correct intervals and out-of-range docs are dropped."""
        from datetime import datetime, timezone
        from types_ import Document

        docs = [
            Document(doc_id=0, keywords=frozenset({"a"}), timestamp=datetime(1999, 11, 28, 12, 0, tzinfo=timezone.utc)), # before start (dropped)
            Document(doc_id=1, keywords=frozenset({"b"}), timestamp=datetime(1999, 12, 1, 0, 0, tzinfo=timezone.utc)),   # month 0
            Document(doc_id=2, keywords=frozenset({"c"}), timestamp=datetime(2000, 1, 15, 10, 0, tzinfo=timezone.utc)),  # month 1
            Document(doc_id=3, keywords=frozenset({"d"}), timestamp=datetime(2000, 6, 20, 15, 0, tzinfo=timezone.utc)),  # month 6
            Document(doc_id=4, keywords=frozenset({"e"}), timestamp=datetime(2001, 2, 27, 23, 0, tzinfo=timezone.utc)),  # month 14
            Document(doc_id=5, keywords=frozenset({"f"}), timestamp=datetime(2001, 3, 1, 8, 0, tzinfo=timezone.utc)),   # after end (dropped)
        ]

        filtered = preprocess.split_into_intervals(
            docs,
            start="1999-12-01",
            end="2001-02-27",
            freq="month",
        )

        assert len(filtered) == 4
        assert [d.doc_id for d in filtered] == [1, 2, 3, 4]
        assert [d.interval for d in filtered] == [0, 1, 6, 14]

    def test_lucene_default_interval_splitting(self):
        """Test Lucene default interval range 2006-06-01 to 2006-11-30."""
        from datetime import datetime, timezone
        from types_ import Document

        docs = [
            Document(doc_id=0, keywords=frozenset({"x"}), timestamp=datetime(2006, 5, 31, 23, 59, tzinfo=timezone.utc)), # dropped
            Document(doc_id=1, keywords=frozenset({"x"}), timestamp=datetime(2006, 6, 1, 0, 0, tzinfo=timezone.utc)),   # month 0
            Document(doc_id=2, keywords=frozenset({"y"}), timestamp=datetime(2006, 8, 15, 12, 0, tzinfo=timezone.utc)),  # month 2
            Document(doc_id=3, keywords=frozenset({"z"}), timestamp=datetime(2006, 11, 30, 23, 59, tzinfo=timezone.utc)), # month 5
            Document(doc_id=4, keywords=frozenset({"w"}), timestamp=datetime(2006, 12, 1, 0, 0, tzinfo=timezone.utc)),  # dropped
        ]

        filtered = preprocess.split_into_intervals(
            docs,
            start="2006-06-01",
            end="2006-11-30",
            freq="month",
        )

        assert len(filtered) == 3
        assert [d.doc_id for d in filtered] == [1, 2, 3]
        assert [d.interval for d in filtered] == [0, 2, 5]

    def test_validation_errors(self):
        """Test invalid start/end dates and unsupported frequency."""
        from types_ import Document
        from datetime import datetime, timezone

        docs = [Document(doc_id=0, keywords=frozenset(), timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc))]

        with pytest.raises(ValueError, match="Start date .* cannot be after end date"):
            preprocess.split_into_intervals(docs, start="2001-01-01", end="2000-01-01")

        with pytest.raises(ValueError, match="Unsupported frequency"):
            preprocess.split_into_intervals(docs, start="2000-01-01", end="2000-02-01", freq="century")

        with pytest.raises(ValueError, match="start and end boundaries"):
            preprocess.split_into_intervals(docs, start=None, end="2000-02-01")

    def test_cli_lucene_prints_omitted_year_note(self, tmp_path: Path):
        """Test CLI prints note when preprocessing Lucene with default dates."""
        mbox_dir = create_fake_mbox(tmp_path)
        out_dir = tmp_path / "cli_lucene"

        cmd = [
            sys.executable,
            "preprocess.py",
            "--source",
            "lucene",
            "--path",
            str(mbox_dir),
            "--out-dir",
            str(out_dir),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        assert "The paper omits the year for the Lucene end date" in result.stdout
        assert "2006-11-30" in result.stdout

    def test_cli_enron_custom_start_end(self, tmp_path: Path):
        """Test CLI with explicit --start and --end options."""
        maildir = create_fake_maildir(tmp_path)
        out_dir = tmp_path / "cli_enron_custom"

        cmd = [
            sys.executable,
            "preprocess.py",
            "--source",
            "enron",
            "--path",
            str(maildir),
            "--start",
            "2001-01-01",
            "--end",
            "2001-01-31",
            "--out-dir",
            str(out_dir),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        assert (out_dir / "enron.pkl").exists()
        with open(out_dir / "enron.pkl", "rb") as f:
            docs, _, _ = pickle.load(f)

        # In January 2001 with start=2001-01-01, month offset is 0
        assert len(docs) == 3
        for d in docs:
            assert d.interval == 0

