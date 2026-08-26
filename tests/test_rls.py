import pytest
from sqlalchemy.exc import ProgrammingError

from knowledge_grove import crud

from conftest import vec


def test_owner_can_see_own_document(alice):
    doc = crud.add_document(
        alice, content="alice's doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert crud.get_by_id(alice, doc.id) is not None


def test_other_agent_cannot_see_ungranted_document(alice, bob):
    doc = crud.add_document(
        alice, content="alice's private doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    assert crud.get_by_id(bob, doc.id) is None


def test_grantee_can_see_document_after_read_grant(alice, bob):
    doc = crud.add_document(
        alice, content="shared doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.grant_access(alice, doc.id, "agent_bob", "read")
    alice.commit()

    seen = crud.get_by_id(bob, doc.id)
    assert seen is not None
    assert seen.id == doc.id


def test_grantee_loses_access_after_revoke(alice, bob):
    doc = crud.add_document(
        alice, content="shared doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.grant_access(alice, doc.id, "agent_bob", "read")
    alice.commit()
    assert crud.get_by_id(bob, doc.id) is not None

    crud.revoke_access(alice, doc.id, "agent_bob")
    alice.commit()

    assert crud.get_by_id(bob, doc.id) is None


def test_read_only_grantee_cannot_write(alice, bob):
    doc = crud.add_document(
        alice, content="shared doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.grant_access(alice, doc.id, "agent_bob", "read")
    alice.commit()

    with pytest.raises(ProgrammingError):
        crud.add_tag(bob, doc.id, "bobs-tag", "bob trying to tag alice's doc")
        bob.commit()
    bob.rollback()


def test_write_grantee_can_write(alice, bob):
    doc = crud.add_document(
        alice, content="shared doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()
    crud.grant_access(alice, doc.id, "agent_bob", "write")
    alice.commit()

    tag = crud.add_tag(bob, doc.id, "bobs-tag", "bob has write access")
    bob.commit()

    assert tag.tag == "bobs-tag"


def test_agent_cannot_grant_access_to_documents_they_do_not_own(alice, bob):
    doc = crud.add_document(
        alice, content="alice's doc", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    with pytest.raises(ProgrammingError):
        crud.grant_access(bob, doc.id, "agent_bob", "read")
        bob.commit()
    bob.rollback()


def test_agent_cannot_insert_document_owned_by_someone_else(alice):
    with pytest.raises(ProgrammingError):
        crud.add_document(
            alice,
            content="pretending to be bob",
            embedding=vec(0),
            owner_agent="agent_bob",
        )
        alice.commit()
    alice.rollback()


def test_owner_can_update_own_document(alice):
    doc = crud.add_document(
        alice, content="v1", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    revised = crud.update_document(alice, doc.id, content="v2", embedding=vec(1))
    alice.commit()

    assert revised.content == "v2"


def test_other_agent_cannot_update_ungranted_document(alice, bob):
    doc = crud.add_document(
        alice, content="v1", embedding=vec(0), owner_agent="agent_alice"
    )
    alice.commit()

    with pytest.raises(ValueError):
        # update_document itself does a get_by_id first; under RLS bob's
        # session can't even see the row, so this fails as "not found"
        # rather than a permission error at the SQL layer.
        crud.update_document(bob, doc.id, content="v2", embedding=vec(1))
