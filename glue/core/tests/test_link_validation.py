# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

"""
Tests for link validation functionality.
"""

import warnings
import pytest
import numpy as np

from glue.core import Data, DataCollection, ComponentID
from glue.core.component_link import ComponentLink
from glue.core.link_helpers import (
    LinkSame, LinkTwoWay, MultiLink, JoinLink, LinkCollection,
    validate_link, LinkValidationError
)
from glue.core.state import GlueSerializer, GlueUnSerializer


# =============================================================================
# Test validate_link() function - Structural validation
# =============================================================================

class TestValidateLinkStructural:
    """Tests for structural validation of links."""

    def setup_method(self):
        self.data1 = Data(x=[1, 2, 3], y=[4, 5, 6], label='data1')
        self.data2 = Data(a=[7, 8, 9], b=[10, 11, 12], label='data2')

    def test_valid_component_link(self):
        """Test that a valid ComponentLink passes validation."""
        link = ComponentLink([self.data1.id['x']], self.data2.id['a'])
        result = validate_link(link)
        assert result is True

    def test_valid_component_link_with_function(self):
        """Test that a valid ComponentLink with transformation passes."""
        link = ComponentLink([self.data1.id['x']], self.data2.id['a'],
                             using=lambda x: x * 2)
        result = validate_link(link)
        assert result is True

    def test_valid_component_link_with_inverse(self):
        """Test that a valid ComponentLink with inverse passes."""
        link = ComponentLink([self.data1.id['x']], self.data2.id['a'],
                             using=lambda x: x * 2,
                             inverse=lambda x: x / 2)
        result = validate_link(link)
        assert result is True

    def test_valid_link_same(self):
        """Test that a valid LinkSame passes validation."""
        link = LinkSame(self.data1.id['x'], self.data2.id['a'])
        result = validate_link(link)
        assert result is True

    def test_valid_link_two_way(self):
        """Test that a valid LinkTwoWay passes validation."""
        link = LinkTwoWay(self.data1.id['x'], self.data2.id['a'],
                          forwards=lambda x: x * 2,
                          backwards=lambda x: x / 2)
        result = validate_link(link)
        assert result is True

    def test_invalid_type_raises_error(self):
        """Test that non-link types raise validation error."""
        with pytest.raises(LinkValidationError, match="Expected ComponentLink"):
            validate_link("not a link")

    def test_invalid_type_no_raise(self):
        """Test that non-link types return error in no-raise mode."""
        is_valid, errors = validate_link("not a link", raise_on_error=False)
        assert is_valid is False
        assert len(errors) == 1
        assert "Expected ComponentLink" in errors[0]


class TestValidateLinkNoRaiseMode:
    """Tests for validate_link with raise_on_error=False."""

    def setup_method(self):
        self.data1 = Data(x=[1, 2, 3], label='data1')
        self.data2 = Data(a=[7, 8, 9], label='data2')

    def test_valid_link_returns_true_empty_errors(self):
        """Test that valid link returns (True, [])."""
        link = ComponentLink([self.data1.id['x']], self.data2.id['a'])
        is_valid, errors = validate_link(link, raise_on_error=False)
        assert is_valid is True
        assert errors == []

    def test_invalid_link_returns_false_with_errors(self):
        """Test that invalid input returns (False, [errors])."""
        is_valid, errors = validate_link(None, raise_on_error=False)
        assert is_valid is False
        assert len(errors) > 0


# =============================================================================
# Test validate_link() - Contextual validation with DataCollection
# =============================================================================

class TestValidateLinkContextual:
    """Tests for contextual validation against DataCollection."""

    def setup_method(self):
        self.data1 = Data(x=[1, 2, 3], y=[4, 5, 6], label='data1')
        self.data2 = Data(a=[7, 8, 9], b=[10, 11, 12], label='data2')
        self.data3 = Data(c=[1, 2, 3], label='data3')  # Not in collection
        self.dc = DataCollection([self.data1, self.data2])

    def test_valid_link_in_collection(self):
        """Test link validation with all CIDs in collection."""
        link = ComponentLink([self.data1.id['x']], self.data2.id['a'])
        result = validate_link(link, self.dc)
        assert result is True

    def test_link_with_from_cid_not_in_collection(self):
        """Test that link with from_cid not in collection fails."""
        link = ComponentLink([self.data3.id['c']], self.data1.id['x'])

        with pytest.raises(LinkValidationError, match="not found in DataCollection"):
            validate_link(link, self.dc)

    def test_link_with_from_cid_not_in_collection_no_raise(self):
        """Test no-raise mode for from_cid not in collection."""
        link = ComponentLink([self.data3.id['c']], self.data1.id['x'])
        is_valid, errors = validate_link(link, self.dc, raise_on_error=False)
        assert is_valid is False
        assert any("not found in DataCollection" in e for e in errors)

    def test_link_with_to_parent_not_in_collection(self):
        """Test that link with to_id parent not in collection fails."""
        link = ComponentLink([self.data1.id['x']], self.data3.id['c'])

        with pytest.raises(LinkValidationError, match="not found in DataCollection"):
            validate_link(link, self.dc)

    def test_link_same_in_collection(self):
        """Test LinkSame validation with DataCollection."""
        link = LinkSame(self.data1.id['x'], self.data2.id['a'])
        result = validate_link(link, self.dc)
        assert result is True


# =============================================================================
# Test LinkEditorState validation integration
# =============================================================================

class TestLinkEditorValidation:
    """Tests for validation integration in LinkEditorState."""

    def setup_method(self):
        self.data1 = Data(x=[1, 2, 3], y=[4, 5, 6], label='data1')
        self.data2 = Data(a=[7, 8, 9], b=[10, 11, 12], label='data2')
        self.dc = DataCollection([self.data1, self.data2])

    def test_update_links_validates(self):
        """Test that update_links_in_collection validates links."""
        from glue.dialogs.link_editor.state import LinkEditorState

        state = LinkEditorState(self.dc)
        state.data1 = self.data1
        state.data2 = self.data2
        state.att1 = self.data1.id['x']
        state.att2 = self.data2.id['a']
        state.simple_link()

        # Should work without errors
        invalid_links = state.update_links_in_collection()
        assert len(invalid_links) == 0

    def test_validate_current_link(self):
        """Test validate_current_link method."""
        from glue.dialogs.link_editor.state import LinkEditorState

        state = LinkEditorState(self.dc)
        state.data1 = self.data1
        state.data2 = self.data2
        state.att1 = self.data1.id['x']
        state.att2 = self.data2.id['a']
        state.simple_link()

        is_valid, errors = state.validate_current_link()
        assert is_valid is True
        assert errors == []

    def test_validate_all_links(self):
        """Test validate_all_links method."""
        from glue.dialogs.link_editor.state import LinkEditorState

        state = LinkEditorState(self.dc)
        state.data1 = self.data1
        state.data2 = self.data2
        state.att1 = self.data1.id['x']
        state.att2 = self.data2.id['a']
        state.simple_link()

        results = state.validate_all_links()
        assert len(results) == 1
        link_state, is_valid, errors = results[0]
        assert is_valid is True


# =============================================================================
# Test session loading with validation
# =============================================================================

class TestSessionLoadValidation:
    """Tests for link validation during session loading."""

    def test_valid_links_preserved_in_session(self):
        """Test that valid links are preserved through save/load cycle."""
        data1 = Data(x=[1, 2, 3], label='data1')
        data2 = Data(a=[7, 8, 9], label='data2')
        dc = DataCollection([data1, data2])

        link = LinkSame(data1.id['x'], data2.id['a'])
        dc.add_link(link)

        # Serialize
        serializer = GlueSerializer(dc)
        state = serializer.dumps()

        # Deserialize
        unserializer = GlueUnSerializer.loads(state)
        dc2 = unserializer.object('__main__')

        # Check link was restored
        assert len(dc2.external_links) == 1

    def test_session_load_with_component_link(self):
        """Test that ComponentLink is preserved through save/load."""
        data1 = Data(x=[1, 2, 3], label='data1')
        data2 = Data(a=[7, 8, 9], label='data2')
        dc = DataCollection([data1, data2])

        link = ComponentLink([data1.id['x']], data2.id['a'])
        dc.add_link(link)

        # Serialize
        serializer = GlueSerializer(dc)
        state = serializer.dumps()

        # Deserialize
        unserializer = GlueUnSerializer.loads(state)
        dc2 = unserializer.object('__main__')

        # Check link was restored
        assert len(dc2.external_links) == 1


# =============================================================================
# Test edge cases
# =============================================================================

class TestValidateLinkEdgeCases:
    """Tests for edge cases in link validation."""

    def setup_method(self):
        self.data1 = Data(x=[1, 2, 3], y=[4, 5, 6], label='data1')
        self.data2 = Data(a=[7, 8, 9], label='data2')

    def test_empty_link_collection(self):
        """Test validation of empty LinkCollection."""
        # LinkAligned with same shape creates links, but we can test
        # the base LinkCollection which might be empty
        class EmptyLinkCollection(LinkCollection):
            pass

        link = EmptyLinkCollection(data1=self.data1, data2=self.data2)
        # Empty collection should still be valid structurally
        is_valid, errors = validate_link(link, raise_on_error=False)
        assert is_valid is True

    def test_link_with_none_parent(self):
        """Test link with ComponentID that has no parent."""
        orphan_cid = ComponentID('orphan')
        link = ComponentLink([self.data1.id['x']], orphan_cid)

        # Structural validation should pass (orphan_cid is valid ComponentID)
        is_valid, errors = validate_link(link, raise_on_error=False)
        assert is_valid is True

    def test_link_with_none_parent_in_collection(self):
        """Test link with orphan ComponentID against DataCollection."""
        dc = DataCollection([self.data1, self.data2])
        orphan_cid = ComponentID('orphan')
        # orphan_cid has no parent, so parent check should pass (parent is None)
        link = ComponentLink([self.data1.id['x']], orphan_cid)

        is_valid, errors = validate_link(link, dc, raise_on_error=False)
        # Should be valid because orphan_cid.parent is None, so we don't check
        assert is_valid is True

    def test_multiple_from_ids(self):
        """Test link with multiple from ComponentIDs."""
        def combine(x, y):
            return x + y

        link = ComponentLink([self.data1.id['x'], self.data1.id['y']],
                             self.data2.id['a'], using=combine)
        is_valid, errors = validate_link(link, raise_on_error=False)
        assert is_valid is True


class TestJoinLinkValidation:
    """Tests for JoinLink validation."""

    def setup_method(self):
        self.data1 = Data(id1=[1, 2, 3], value1=[10, 20, 30], label='data1')
        self.data2 = Data(id2=[2, 3, 4], value2=[200, 300, 400], label='data2')
        self.dc = DataCollection([self.data1, self.data2])

    def test_valid_join_link(self):
        """Test that valid JoinLink passes validation."""
        link = JoinLink(cids1=[self.data1.id['id1']],
                        cids2=[self.data2.id['id2']],
                        data1=self.data1,
                        data2=self.data2)
        result = validate_link(link)
        assert result is True

    def test_join_link_in_collection(self):
        """Test JoinLink validation with DataCollection."""
        link = JoinLink(cids1=[self.data1.id['id1']],
                        cids2=[self.data2.id['id2']],
                        data1=self.data1,
                        data2=self.data2)
        result = validate_link(link, self.dc)
        assert result is True
